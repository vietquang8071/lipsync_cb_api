# Đề xuất cải thiện tốc độ phản hồi của `flash_head`

## Kết luận ngắn

Đường chạy nhanh nên ưu tiên `lite` (hoặc `pro`) với 4 bước sampling, giữ pipeline sống lâu và warm-up trước khi nhận request. Không nên dùng `pretrained` cho luồng tương tác: cấu hình hiện tại chạy 20 bước và mỗi bước gọi DiT hai lần để làm classifier-free guidance, tức khoảng **40 lần forward DiT/chunk**, trong khi `lite`/`pro` chỉ gọi **4 lần/chunk**.

Bốn cơ hội lớn nhất trong code hiện tại là:

1. Luồng streaming mã hóa lại toàn bộ cửa sổ audio 8 giây bằng Wav2Vec2 ở mỗi chunk mới, dù cuối cùng chỉ lấy 33 vị trí gần cuối.
2. Phép chiếu audio và K/V của cross-attention không phụ thuộc timestep nhưng đang bị tính lại trong mọi bước denoise.
3. `torch.compile` được bật mặc định nhưng không có warm-up/persistent cache rõ ràng, nên request đầu tiên có thể chịu cả thời gian compile.
4. Kết quả 33 frame 512×512 được đổi sang `float32` rồi mới chuyển về CPU; riêng tensor này xấp xỉ 104 MB/chunk, gấp 4 lần dữ liệu `uint8` thực sự cần cho video.

Đây là phân tích tĩnh dựa trên source; chưa có checkpoint/runtime đúng để benchmark nên không gán số tăng tốc tuyệt đối. Cần đo từng stage bằng CUDA Event trước khi chốt.

## Đường găng hiện tại

```text
audio -> feature extractor -> Wav2Vec2 -> tạo cửa sổ audio
      -> 4/20 bước denoise DiT -> VAE decode 33 frame
      -> Lab color correction toàn chunk -> VAE encode frame nối tiếp
      -> đổi layout + float32 0..255 -> CPU/video encoder
```

Các vị trí chính:

- Cấu hình 33 frame, 512×512, cache audio 8 giây: `flash_head/configs/infer_params.yaml:1-10`.
- Số bước 20 cho `pretrained`, 4 cho model còn lại: `flash_head/inference.py:32-36`.
- Compile model/VAE: `flash_head/src/pipeline/flash_head_pipeline.py:129-137`.
- Wav2Vec2 và lấy toàn bộ 12 hidden states: `flash_head/src/pipeline/flash_head_pipeline.py:206-226`.
- Denoise, decode, color correction và encode lại motion frame: `flash_head/src/pipeline/flash_head_pipeline.py:229-316`.
- Audio projection nằm bên trong mỗi `model.forward`: `flash_head/src/modules/flash_head_model.py:429-471`.
- RoPE được dựng lại trong từng block: `flash_head/src/modules/flash_head_model.py:104-140`.

## Đề xuất theo mức ưu tiên

| Ưu tiên | Đề xuất | Tác động dự kiến | Rủi ro chất lượng |
|---|---|---:|---:|
| P0 | Giữ pipeline thường trú, compile/warm-up trước khi mở traffic | Rất lớn với cold start | Thấp |
| P0 | Không mã hóa lại 8 giây audio ở mỗi chunk | Lớn | Trung bình nếu rút ngắn context |
| P0 | Cache audio projection và K/V cross-attention theo chunk | Lớn, nhất là model 20 bước | Thấp nếu cache đúng |
| P0 | Bỏ `cuda.synchronize()` và `print` khỏi hot path | Vừa | Thấp |
| P0 | Trả frame `uint8`, copy bất đồng bộ hoặc encode trực tiếp | Vừa | Thấp |
| P1 | Giảm chunk từ 33 xuống 17 frame và giữ shape cố định | Lớn, giảm cả TTFF | Trung bình/cao |
| P1 | Tránh VAE encode lại motion frame khi có thể tái sử dụng latent | Vừa/lớn | Trung bình |
| P1 | Tối ưu hoặc tắt color correction theo profile | Vừa | Trung bình |
| P1 | Cache RoPE, bỏ FP64 và benchmark backend attention | Vừa | Thấp/trung bình |
| P1 | Cache/lazy-load latent của ảnh tham chiếu | Lớn với request/avatar đầu tiên | Thấp |
| P2 | CUDA Graph, FP8/INT8/weight-only quantization | Lớn sau khi đã ổn định shape | Trung bình/cao |
| P2 | Multi-GPU USP theo benchmark, không mặc định coi là nhanh hơn | Tùy model | Thấp |

## 1. Tách cold start khỏi warm latency

`torch.compile` chỉ bọc model tại lúc khởi tạo; chi phí compile thật phát sinh khi lần đầu gặp input. Vì vậy request đầu tiên có thể đồng thời chịu:

- load checkpoint và chuyển model lên GPU;
- compile VAE encode khi chuẩn bị ảnh;
- compile DiT và VAE decode khi tạo chunk đầu;
- khởi tạo kernel/backend attention.

Đề xuất:

- Dùng worker/pipeline sống lâu, không load lại theo request.
- Warm-up đầy đủ đúng từng profile cố định `(model_type, frame_num, H, W, dtype)` trước khi báo healthy.
- Sau warm-up phải reset seed, `latent_motion_frames` và state phiên, vì `generate()` có thay đổi state.
- Bật persistent Inductor cache giữa các lần restart nếu phiên bản PyTorch/GPU không đổi.
- Nếu đây là CLI one-shot và cold latency quan trọng hơn tổng thời gian, benchmark cả `COMPILE_MODEL=False`; compile chỉ có lợi khi đủ nhiều chunk để hoàn vốn.
- Theo dõi số graph/recompile. Shape audio hoặc frame thay đổi có thể làm mất lợi ích compile.

## 2. Loại bỏ tính toán audio lặp lại

Trong streaming, `cached_audio_duration=8` tạo input khoảng 200 vị trí ở 25 fps. Mỗi chunk chỉ thêm khoảng 24 frame đối với Lite hoặc 28 frame đối với Pro theo cách caller hiện tại, nhưng `preprocess_audio()` vẫn chạy lại feature extractor và toàn bộ Wav2Vec2 cho cả 8 giây. Sau đó `get_audio_embedding()` mới cắt ra 33 tâm và cửa sổ 5 vị trí.

Thứ tự thử nghiệm an toàn:

1. Với file audio có sẵn, luôn ưu tiên encode một lần rồi cắt embedding theo chunk.
2. Với audio thật sự streaming, cache waveform/feature CNN và chỉ tính phần mới cộng receptive-field overlap.
3. Wav2Vec2 encoder là hai chiều nên KV-cache chính xác không đơn giản. Thử giảm context từ 8 giây xuống 2-4 giây bằng A/B test; đây thường là cách thực dụng hơn.
4. Nếu bắt buộc giữ đúng context 8 giây, cân nhắc encoder causal/streaming hoặc distill audio encoder; đây là thay đổi model, không còn là tối ưu thuần runtime.
5. Wav2Vec2 hiện chạy FP32 (`.to(self.device)` không đổi dtype). Benchmark autocast BF16/FP16, nhưng giữ LayerNorm/những op nhạy cảm ở FP32 nếu cần.

Không thể đơn giản tắt `output_hidden_states=True`, vì `AudioProjModel` đang dùng đủ 12 layer. Muốn chỉ dùng vài layer cần huấn luyện/fine-tune lại adapter.

Ngoài ra, tensor chỉ mục cửa sổ ở `flash_head/inference.py:64-69` có thể cache sẵn trên GPU cho mỗi `frame_num`, tránh tạo `arange`, clamp và chỉ mục CPU lặp lại.

## 3. Cache phần conditioning không đổi giữa các timestep

Trong một chunk, `audio_embedding` không đổi qua các bước denoise, nhưng mỗi lần `WanModelAudioProject.forward()` đều làm lại:

- sắp xếp/cắt cửa sổ audio;
- `AudioProjModel`, gồm các linear projection lớn;
- trong từng DiT block, linear K/V của cross-attention trên cùng context.

Nên tách model thành hai pha:

- `prepare_condition(audio_embedding)`: tạo projected audio và K/V riêng cho từng block một lần/chunk;
- `denoise_step(x, timestep, prepared_condition)`: mỗi bước chỉ tính Q và phần phụ thuộc latent/timestep.

Với Lite/Pro 4 bước, riêng `AudioProjModel` có thể giảm từ 4 lần xuống 1 lần. Với `pretrained`, hai nhánh conditional/unconditional hiện tạo 40 lần forward; chỉ cần cache hai bộ conditioning, thay vì tính lại 40 lần. `torch.zeros_like(audio_embedding)` của nhánh unconditional cũng nên tạo một lần ngoài vòng lặp.

Đây là tối ưu ít rủi ro chất lượng nhất vì biểu thức toán học không thay đổi; cần so sánh output với baseline ở cùng seed để xác nhận sai số chỉ do dtype/kernel.

## 4. Gỡ các điểm đồng bộ CPU/GPU khỏi hot path

`generate()` gọi `torch.cuda.synchronize()` trước và sau từng bước denoise, rồi tiếp tục đồng bộ quanh decode, color correction và encode motion. Các lệnh này phục vụ đo thời gian nhưng làm CPU chờ GPU và chặn khả năng overlap. `print()` ở mỗi bước cũng thêm I/O đồng bộ.

Đề xuất:

- Production path không synchronize theo step và không print theo step.
- Khi profiling, dùng CUDA Event trên stream; chỉ synchronize một lần ở cuối khoảng đo.
- Log metric gộp theo chunk hoặc sampling, không ghi stdout trong vòng denoise.
- Với multi-GPU, chỉ giữ barrier/all-gather thật sự cần cho đúng dữ liệu.

Việc bỏ synchronize có thể không giảm nhiều thời gian kernel thuần vì các bước denoise phụ thuộc tuần tự, nhưng sẽ giảm launch/I/O overhead và cho phép CPU chuẩn bị/copy/encode song song.

## 5. Giảm độ trễ và băng thông đầu ra

`FlashHeadPipeline.generate()` ép video sang FP32; `run_pipeline()` tiếp tục scale sang 0..255 nhưng không cast `uint8`. Với 33×512×512×3, payload FP32 khoảng 104 MB/chunk, trong khi `uint8` khoảng 26 MB.

Đề xuất:

- Giữ BF16/FP16 sau VAE, clamp/scale rồi cast thẳng `uint8` trên GPU.
- Chỉ chuyển các frame thực sự trả về; caller đang bỏ các motion frame đầu nhưng `run_pipeline()` vẫn permute/scale toàn bộ 33 frame.
- Dùng pinned host buffer và non-blocking copy, overlap copy chunk N với generate chunk N+1.
- Nếu kiến trúc phục vụ cho phép, đưa frame thẳng sang NVENC/GStreamer thay vì GPU -> tensor CPU float -> NumPy -> encoder.
- Với WanVAE, source đã có `decode_stream()` nhưng pipeline chưa dùng. Có thể yield từng nhóm frame sau denoise để giảm time-to-first-frame, rồi color-correct từng frame/nhóm. Cách này giảm TTFF sau denoise, không nhất thiết giảm tổng compute.

## 6. Giảm kích thước chunk và độ phân giải theo profile

`frame_num=33` ở 25 fps tương ứng 1,32 giây video được tạo mỗi lần. Transformer self-attention tăng rất nhanh theo số token không-thời gian. Profile tương tác nên thử:

- `frame_num=17` vì `17 - 1` vẫn chia hết cho temporal stride 4 hoặc 8 thường dùng ở hai VAE;
- 384×384 hoặc 448×448 nếu checkpoint cho phép;
- giữ một số ít shape cố định để tránh `torch.compile` recompile.

Không thay đổi trực tiếp cấu hình production trước khi kiểm tra:

- lip-sync (LSE-C/LSE-D hoặc SyncNet confidence);
- identity similarity;
- rung/giật tại biên chunk;
- màu da và chuyển động đầu;
- VRAM, p50/p95 chunk latency và real-time factor.

Giảm 33 xuống 17 frame không chỉ giảm thời gian chờ gom audio mà còn giảm temporal token gần một nửa; phần self-attention có thể giảm mạnh hơn tuyến tính. Đổi lại, tỷ lệ frame overlap tăng và biên chunk dễ lộ hơn.

## 7. Tránh decode rồi encode lại chỉ để giữ continuity

Cuối mỗi chunk, code VAE-decode toàn bộ latent, color-correct ở pixel space, lấy các frame cuối rồi VAE-encode lại thành `latent_motion_frames`. Đây là một vòng decode -> encode đắt.

Ba phương án A/B theo độ rủi ro tăng dần:

1. Giữ hiện trạng làm baseline chất lượng.
2. Nếu tắt color correction, thử lấy trực tiếp các latent cuối từ kết quả denoise làm motion state, bỏ VAE encode cuối chunk.
3. Nếu vẫn cần color correction, chỉ encode đúng phần overlap và thử cache state temporal của VAE giữa các chunk.

Phương án lấy latent trực tiếp phải kiểm tra seam/độ trôi danh tính. Pixel đã color-correct không còn tương đương latent trước decode nên không nên mặc định coi hai đường là giống nhau.

## 8. Tối ưu color correction

Mặc định strength là 1.0, vì vậy mỗi chunk chạy RGB -> Lab -> thống kê -> Lab -> RGB trên toàn bộ video full-resolution. Code còn tính lại Lab mean/std của cùng ảnh tham chiếu ở mỗi chunk và tạo lại các ma trận chuyển màu.

Đề xuất:

- Cache Lab mean/std của ảnh tham chiếu khi `prepare_params()`.
- Cache các hằng số/ma trận chuyển màu theo `(device, dtype)`.
- Trong chế độ streaming, không color-correct những motion frame đầu chắc chắn bị loại khỏi output.
- Benchmark thuật toán đơn giản hơn trong RGB/YUV hoặc LUT/fused Triton kernel.
- Cung cấp profile `fast` với strength 0 hoặc thấp hơn nếu đánh giá hình ảnh chấp nhận được.

## 9. Tối ưu RoPE, norm và attention kernel

`rope_apply()` hiện tạo lại lưới frequency cho Q và K trong mọi self-attention block, đồng thời chuyển Q/K sang `float64` và dùng complex arithmetic. Trên RTX 4090, FP64 không phải đường tính toán tối ưu.

Đề xuất:

- Cache cos/sin hoặc complex frequency đã broadcast theo `(F,H,W,device,SP rank)` một lần cho mỗi shape.
- Dùng FP32/complex64 cho RoPE, trả về dtype model sau phép quay; xác nhận sai số bằng output cùng seed.
- Dùng fused RMSNorm/LayerNorm nếu profiler cho thấy đáng kể.
- Log rõ backend attention được chọn. Code ưu tiên SageAttention, rồi FlashAttention 3/2, rồi PyTorch SDPA; benchmark trên đúng GPU/shape thay vì giả định package nào cũng nhanh nhất.
- Với shape cố định, đánh giá CUDA Graph sau khi `torch.compile` đã ổn định.

## 10. Cache ảnh/avatar và chuẩn bị lười

`prepare_params()` VAE-encode một video tĩnh 33 frame cho **mọi ảnh** trong thư mục trước khi chọn người đầu tiên. Nếu có nhiều avatar hoặc cùng avatar được dùng lại giữa request, thời gian phản hồi đầu sẽ tăng không cần thiết.

Đề xuất:

- Cache `cond_image_tensor`, `ref_img_latent` và thống kê màu theo hash ảnh + model/VAE + resolution.
- Chỉ encode avatar đang được chọn; avatar khác encode khi `reset_person_name()` lần đầu.
- Face detection chỉ nằm ở pha chuẩn bị và được ghi chú khoảng 2 ms/frame, nên không phải mục tiêu tối ưu đầu tiên.

## 11. Quantization và phần cứng

Sau khi các tính toán lặp đã được loại bỏ mới nên thử:

- FP8 cho linear/MLP trên Ada, hoặc weight-only INT8/INT4;
- giữ norm, softmax, VAE output và các op nhạy cảm ở BF16/FP32;
- calibration trên nhiều người nói, âm vị, màu da và chuyển động đầu;
- TensorRT/torch-tensorrt chỉ khi graph/shape đã ổn định và tỷ lệ fallback thấp.

Multi-GPU USP có thể giúp Pro ở sequence dài, nhưng communication/all-gather có thể khiến Lite batch 1 chậm hơn một GPU. Cần đo 1 GPU và 2 GPU bằng cùng workload; không tối ưu theo throughput nếu mục tiêu là latency một request.

Pipeline hiện có state mutable (`generator`, `person_name`, `latent_motion_frames`), nên không an toàn để nhiều phiên đồng thời dùng chung một instance. Với tải nhiều người dùng, cần tách state theo session hoặc pool worker; micro-batching cải thiện throughput nhưng thường làm xấu TTFF.

## Kế hoạch benchmark đề xuất

Đo riêng cold và warm path, tối thiểu các stage sau:

1. load model;
2. compile/warm-up;
3. face/ảnh tham chiếu + VAE encode;
4. feature extractor + Wav2Vec2;
5. từng DiT step và tổng denoise;
6. VAE decode;
7. color correction;
8. VAE encode motion;
9. layout/cast + D2H copy;
10. video encode nếu nó nằm trong latency người dùng thấy.

Quy trình đo:

- 2-3 lượt warm-up, sau đó ít nhất 30 chunk cùng shape;
- CUDA Event cho GPU stage, wall clock cho end-to-end;
- báo p50, p95, peak VRAM, time-to-first-frame và real-time factor;
- mục tiêu streaming là p95 compute của một chunk nhỏ hơn thời lượng audio mới của chunk, có thêm headroom cho copy/encode;
- mỗi tối ưu phải chạy cùng seed và cùng tập audio/avatar để so chất lượng.

Ma trận thử nghiệm nên đi tuần tự, không thay nhiều biến cùng lúc:

1. Lite 4-step, 33 frame, 512, baseline.
2. Bỏ synchronize/log + warm compile.
3. Cache audio projection/KV.
4. Audio context 8 -> 4 -> 2 giây.
5. FP32 output -> uint8 output.
6. Color correction 1.0 -> 0.5 -> 0.
7. 33 -> 17 frame.
8. 512 -> 448/384.
9. Attention backend và quantization.

## Thứ tự triển khai khuyến nghị

Nếu mục tiêu là có kết quả nhanh mà ít ảnh hưởng chất lượng:

1. Xác nhận runtime CUDA/attention backend, dựng profiler chuẩn.
2. Pipeline thường trú + compile cache + warm-up.
3. Bỏ synchronize/print khỏi production.
4. Trả `uint8`, copy/encode bất đồng bộ.
5. Cache avatar/color statistics.
6. Cache `AudioProjModel` và cross-attention K/V.
7. Sửa luồng streaming audio để không encode lại 8 giây.
8. Sau đó mới A/B chunk 17 frame, context audio ngắn, bỏ VAE re-encode và quantization.

## Ghi chú về môi trường hiện tại

Trong shell dùng để rà soát, máy có 2 RTX 4090 và driver 535.309.01, nhưng Python hiện tại báo PyTorch 2.13.0+cu130 không khởi tạo được CUDA; đồng thời không thấy FlashAttention/SageAttention. README lại hướng dẫn một conda env Python 3.10 với PyTorch 2.7.1+cu128 và cài hai attention package. Điều này cho thấy shell rà soát có thể không phải env chạy thật.

Cần kiểm tra ngay **bên trong đúng process phục vụ**: `torch.cuda.is_available()`, phiên bản driver/runtime, GPU thực tế và backend attention đã chọn. Nếu production cũng dùng driver/runtime không tương thích thì đây là lỗi môi trường, không phải nút thắt model; mọi benchmark trước khi sửa môi trường đều không có giá trị.

---

Không có file mã nguồn nào được sửa; tài liệu này chỉ đưa ra đề xuất.
