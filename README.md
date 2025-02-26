[![test]([https://tenor.com/vi/view/catspin-gif-10390165497725138728)](https://tenor.com/vi/view/catspin-gif-10390165497725138728](https://media1.tenor.com/m/5ODw4_euPV8AAAAC/the-cat-is-spinning-spin-cat.gif))






# BeatHeritage

Bạn có thể thử model [tại đây](https://colab.research.google.com/github/hongminh54/BeatHeritage/blob/main/colab/beatheritage_inference.ipynb).

BeatHeritage là một khung đa mô hình sử dụng đầu vào từ phổ ký để tạo ra các beatmap osu! hoàn chỉnh cho tất cả các chế độ chơi. Mục tiêu của dự án này là tự động tạo ra các beatmap osu! có chất lượng đủ để xếp hạng từ bất kỳ bài hát nào với mức độ tùy chỉnh cao.

Dự án này được xây dựng dựa trên [osuT5](https://github.com/gyataro/osuT5) và [osu-diffusion](https://github.com/OliBomby/osu-diffusion).

#### Hãy sử dụng công cụ này một cách có trách nhiệm!


## Hướng dẫn

Hướng dẫn dưới đây cho phép bạn tạo beatmap trên máy tính của mình hoặc bạn có thể chạy nó trên đám mây với [Colab Notebook](https://colab.research.google.com/github/hongminh54/BeatHeritage/blob/main/colab/beatheritage_inference.ipynb).

### 1. Sao chép kho lưu trữ

```sh
git clone https://github.com/hongminh54/BeatHeritage.git
cd BeatHeritage
```

### 2. (Tùy chọn) Tạo môi trường ảo

```sh
python -m venv .venv

# Trong cmd.exe
.venv\Scripts\activate.bat
# Trong PowerShell
.venv\Scripts\Activate.ps1
# Trong Linux or MacOS
source .venv/bin/activate
```

### 3. Cài đặt các phụ thuộc

Cài đặt Python 3.10, [Git](https://git-scm.com/downloads), [ffmpeg](http://www.ffmpeg.org/), [PyTorch](https://pytorch.org/get-started/locally/), và cài đặt các phụ thuộc Python còn lại.

```sh
pip install -r requirements.txt
```

### 4. Bắt đầu quá trình suy luận

Chạy `inference.py` và truyền vào một số đối số để tạo beatmaps. Đối với việc này, hãy sử dụng [Hydra override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/). Xem `inference.yaml` để biết tất cả các tham số có sẵn. 
```
python inference.py \
  audio_path           [Đường dẫn đến file âm thanh đầu vào] \
  output_path          [Đường dẫn đến thư mục đầu ra] \
  beatmap_path         [Đường dẫn đến file .osu để tự động điền metadata, audio_path và output_path, hoặc sử dụng làm tham chiếu] \
  
  gamemode             [Chế độ chơi để tạo 0=std, 1=taiko, 2=ctb, 3=mania] \
  difficulty           [Độ khó sao để tạo] \
  mapper_id            [ID người dùng mapper cho phong cách] \
  year                 [Năm upload để mô phỏng] \
  hitsounded           [Có thêm hitsounds hay không] \
  slider_multiplier    [Hệ số nhân tốc độ slider] \
  circle_size          [Kích thước vòng tròn] \
  keycount             [Số phím cho mania] \
  hold_note_ratio      [Tỷ lệ nốt giữ cho mania 0-1] \
  scroll_speed_ratio   [Tỷ lệ tốc độ cuộn cho mania và ctb 0-1] \
  descriptors          [Danh sách các mô tả OMDB cho phong cách] \
  negative_descriptors [Danh sách các mô tả phủ định OMDB để hướng dẫn không phân loại] \
  
  add_to_beatmap       [Có thêm nội dung được tạo vào beatmap tham chiếu thay vì tạo beatmap mới] \
  start_time           [Thời gian bắt đầu tạo tính bằng mili giây] \
  end_time             [Thời gian kết thúc tạo tính bằng mili giây] \
  in_context           [Danh sách các ngữ cảnh bổ sung để cung cấp cho mô hình [NONE,TIMING,KIAI,MAP,GD,NO_HS]] \
  output_type          [Danh sách các loại nội dung để tạo] \
  cfg_scale            [Tỷ lệ hướng dẫn không phân loại] \
  super_timing         [Có sử dụng bộ tạo thời gian BPM biến đổi chậm chính xác không] \
  seed                 [Hạt giống ngẫu nhiên để tạo] \
```

Ví dụ:
```
python inference.py beatmap_path="'C:\Users\USER\AppData\Local\osu!\Songs\1 Kenji Ninuma - DISCO PRINCE\Kenji Ninuma - DISCOPRINCE (peppy) [Normal].osu'" gamemode=0 difficulty=5.5 year=2023 descriptors="['jump aim','clean']" in_context=[TIMING,KIAI]
```

### Mẹo

### Hướng Dẫn Sử Dụng Model

- Bạn có thể chỉnh sửa tệp `configs/inference_v29.yaml` và thêm các tham số của mình vào đó thay vì nhập chúng trong terminal mỗi lần.
- Tất cả các mô tả (descriptors) có sẵn có thể được tìm thấy [tại đây](https://omdb.nyahh.net/descriptors/).
- Luôn cung cấp tham số năm (year) trong khoảng từ 2007 đến 2023. Nếu để trống, model có thể tạo ra kết quả với phong cách không nhất quán.
- Luôn cung cấp tham số độ khó (difficulty). Nếu để trống, model có thể tạo ra độ khó không nhất quán.
- Tăng giá trị tham số `cfg_scale` để tăng hiệu quả của các tham số `mapper_id` và `descriptors`.
- Bạn có thể sử dụng tham số `negative_descriptors` để hướng model tránh xa một số phong cách nhất định. Tham số này chỉ hoạt động khi `cfg_scale > 1`. Đảm bảo số lượng `negative_descriptors` bằng với số lượng `descriptors`.
- Nếu phong cách bài hát và phong cách beatmap mong muốn không phù hợp, model có thể không tạo ra kết quả như ý muốn. Ví dụ, rất khó để tạo một beatmap có độ khó cao (SR cao, SV cao) cho một bài hát nhẹ nhàng.
- Nếu bạn đã có sẵn phần timing và kiai times cho bài hát, bạn có thể cung cấp chúng cho model để tăng tốc độ và độ chính xác của quá trình suy luận bằng cách sử dụng các tham số `beatmap_path` và `in_context=[TIMING,KIAI]`.
- Để remap một phần beatmap, hãy sử dụng các tham số `beatmap_path`, `start_time`, `end_time` và `add_to_beatmap=true`.
- Để tạo độ khó khách mời (guest difficulty) cho một beatmap, hãy sử dụng các tham số `beatmap_path` và `in_context=[GD,TIMING,KIAI]`.
- Để tạo hitsounds cho một beatmap, hãy sử dụng các tham số `beatmap_path` và `in_context=[NO_HS,TIMING,KIAI]`.
- Để chỉ tạo timing cho một bài hát, hãy sử dụng `super_timing=true` và `output_type=[TIMING]`.

## Tổng quan

### Tokenization

BeatHeritage chuyển đổi các beatmap của osu! thành một dạng biểu diễn sự kiện trung gian, có thể được chuyển đổi trực tiếp sang và từ các token.

Nó bao gồm:  
- Hit objects  
- Hitsounds  
- Slider velocities  
- New combos  
- Timing points  
- Kiai times  
- Tốc độ cuộn của Taiko/Mania  

Dưới đây là một ví dụ nhỏ về quá trình mã hóa token:

![BeatHeritage_parser](https://github.com/user-attachments/assets/84efde76-4c27-48a1-b8ce-beceddd9e695)

Để giảm kích thước từ vựng, các sự kiện thời gian được lượng tử hóa thành các khoảng 10ms, và tọa độ vị trí được lượng tử hóa thành lưới 32 pixel

### Kiến trúc mô hình

Mô hình về cơ bản là một lớp bao bọc xung quanh mô hình [HF Transformers Whisper](https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration), với embedding đầu vào và hàm mất mát tùy chỉnh.  
Kích thước mô hình bao gồm 219 triệu tham số.  
Mô hình này được phát hiện nhanh hơn và chính xác hơn so với T5 cho tác vụ này.  

Tổng quan cấp cao về đầu vào - đầu ra của mô hình như sau:  

![Picture2](https://user-images.githubusercontent.com/28675590/201044116-1384ad72-c540-44db-a285-7319dd01caad.svg)  

Mô hình sử dụng các khung Mel spectrogram làm đầu vào cho encoder, với một khung trên mỗi vị trí đầu vào.  
Đầu ra của decoder tại mỗi bước là một phân phối softmax trên một tập hợp từ vựng sự kiện rời rạc được xác định trước.  
Đầu ra có tính thưa (sparse), chỉ cần có sự kiện khi một hit-object xuất hiện, thay vì chú thích từng khung âm thanh riêng lẻ.  


### Định dạng huấn luyện đa nhiệm

![Định dạng huấn luyện đa nhiệm](https://github.com/user-attachments/assets/62f490bc-a567-4671-a7ce-dbcc5f9cd6d9)

Trước token SOS có các token bổ sung giúp tạo dữ liệu có điều kiện.  
Những token này bao gồm chế độ chơi (gamemode), độ khó (difficulty), ID mapper, năm phát hành và các metadata khác.  

Trong quá trình huấn luyện, các token này không có nhãn đi kèm, nên mô hình sẽ không tạo ra chúng.  
Cũng trong quá trình huấn luyện, có một xác suất ngẫu nhiên một token metadata bị thay thế bởi token "unknown", nên khi suy luận (inference), ta có thể sử dụng "unknown" để giảm lượng metadata cần cung cấp cho mô hình.  

### Sinh chuỗi dài liền mạch

Độ dài ngữ cảnh của mô hình là 8.192 giây, không đủ để tạo toàn bộ beatmap, nên bài hát được chia thành nhiều cửa sổ nhỏ để mô hình tạo beatmap theo từng phần.  

Để đảm bảo beatmap không có đường nối lộ rõ giữa các cửa sổ, chúng tôi áp dụng chồng lấp 90% giữa các cửa sổ và sinh từng cửa sổ theo thứ tự liên tiếp.  
Mỗi cửa sổ (trừ cửa sổ đầu tiên) đều có 50% nội dung đầu được điền trước bằng token từ cửa sổ trước đó.  
Bộ xử lý logit đảm bảo mô hình không tạo token thời gian thuộc 50% đầu tiên của cửa sổ.  
40% cuối của cửa sổ được dành riêng cho cửa sổ kế tiếp. Các token thời gian sinh ra trong phạm vi này được coi là EOS.  
Nhờ đó, mỗi token được tạo ra dựa trên ít nhất 4 giây token trước và 3.3 giây âm thanh sau để dự đoán chính xác hơn.  

Để tránh trôi lệch thời gian trong quá trình sinh dài, trong quá trình huấn luyện, mô hình được thêm độ lệch ngẫu nhiên vào các sự kiện thời gian.  
Điều này buộc mô hình phải dựa vào điểm khởi phát âm thanh thay vì dữ liệu thời gian tuyệt đối.  
Kết quả là mô hình tự động điều chỉnh thời gian một cách chính xác và nhất quán.  


## Tinh chỉnh tọa độ bằng khuếch tán (diffusion)

Tọa độ vị trí được tạo bởi decoder được lượng tử hóa thành lưới 32 pixel, vì vậy sau đó sử dụng khuếch tán (diffusion) để khử nhiễu và tinh chỉnh chúng đến vị trí cuối cùng.  
Để làm điều này, chúng tôi đã huấn luyện một phiên bản sửa đổi của [osu-diffusion](https://github.com/OliBomby/osu-diffusion), chuyên biệt hóa chỉ cho 10% cuối của lịch trình nhiễu và chấp nhận các token metadata nâng cao hơn mà BeatHeritage sử dụng để tạo dữ liệu có điều kiện.  

Vì mô hình BeatHeritage xuất SV (slider velocity) của slider, nên độ dài cần thiết của slider được cố định bất kể hình dạng của đường điều khiển (control point path).  
Do đó, tôi cố gắng hướng dẫn quá trình khuếch tán để tạo ra các tọa độ phù hợp với độ dài slider yêu cầu.  
Chúng tôi thực hiện điều này bằng cách tính toán lại vị trí kết thúc của slider sau mỗi bước của quá trình khuếch tán, dựa trên độ dài yêu cầu và đường điều khiển hiện tại.  
Điều này có nghĩa là quá trình khuếch tán không kiểm soát trực tiếp vị trí kết thúc của slider, nhưng vẫn có thể ảnh hưởng đến nó bằng cách thay đổi đường điều khiển.  


### Xử lý hậu kỳ

BeatHeritage thực hiện một số bước xử lý hậu kỳ để cải thiện chất lượng của beatmap được tạo:

- Tinh chỉnh tọa độ vị trí bằng khuếch tán (diffusion).
- Căn chỉnh lại sự kiện thời gian về tick gần nhất bằng cách sử dụng snap divisors do mô hình tạo ra.
- Điều chỉnh các vị trí gần như trùng khớp hoàn hảo.
- Chuyển đổi sự kiện cột trong chế độ mania thành tọa độ X.
- Tạo đường slider cho taiko drumrolls.
- Sửa các chênh lệch lớn giữa độ dài slider yêu cầu và độ dài đường điều khiển.

### Bộ tạo siêu thời gian (Super timing generator)

Super timing generator là một thuật toán cải thiện độ chính xác của thời gian được tạo bằng cách suy luận thời gian cho toàn bộ bài hát 20 lần và tính trung bình kết quả.  
Điều này đặc biệt hữu ích cho các bài hát có BPM thay đổi hoặc có nhiều phân đoạn BPM khác nhau.  
Kết quả gần như hoàn hảo, chỉ đôi khi có một số đoạn cần điều chỉnh thủ công.  


## Xem thêm
- [Mapper Classifier](./classifier/README.md)
- [RComplexion](./rcomplexion/README.md)

## Các dự án liên quan

1. [osu! Beatmap Generator](https://github.com/Syps/osu_beatmap_generator) - Tác giả: Syps (Nick Sypteras)  
2. [osumapper](https://github.com/kotritrona/osumapper) - Tác giả: kotritrona, jyvden, Yoyolick (Ryan Zmuda)  
3. [osu-diffusion](https://github.com/OliBomby/osu-diffusion) - Tác giả: OliBomby (Olivier Schipper), NiceAesth (Andrei Baciu)  
4. [osuT5](https://github.com/gyataro/osuT5) - Tác giả: gyataro (Xiwen Teoh)  
5. [Beat Learning](https://github.com/sedthh/BeatLearning) - Tác giả: sedthh (Richard Nagyfi)  
6. [osu!dreamer](https://github.com/jaswon/osu-dreamer) - Tác giả: jaswon (Jason Won)  
 

 ### Project dựa trên mã nguồn của OliBomby để xây dựng [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator)
