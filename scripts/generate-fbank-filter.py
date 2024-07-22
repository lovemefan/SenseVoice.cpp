#!/usr/bin/env python3
import math


def mel_scale(data):
    return 1127.0 * math.log(1.0 + data / 700.0)


def main():
    n_mel = 80
    fft_bins = 256
    s = "// Auto-generated. Do NOT edit!\n\n"
    s += "\n"

    s += f"const int LogMelFilterRows = {n_mel};\n"
    s += f"const int LogMelFilterCols = {fft_bins};\n"
    s += "\n"
    s += "const float LogMelFilterMelArray[] = {\n\t"
    sep = ""
    mel_low_freq = 31.748642
    mel_hight_freq = 2840.03784
    mel_freq_delta = 34.6702385
    fft_bin_width = 31.25
    count = 0
    for i in range(n_mel):
        left_mel = mel_low_freq + i * mel_freq_delta
        center_mel = mel_low_freq + (i + 1.0) * mel_freq_delta
        right_mel = mel_low_freq + (i + 2.0) * mel_freq_delta

        for j in range(fft_bins):
            mel_num = mel_scale(fft_bin_width * j)
            up_slope = (mel_num - left_mel) / (center_mel - left_mel)
            down_slope = (right_mel - mel_num) / (right_mel - center_mel)
            _filter = max(0.0, min(up_slope, down_slope))
            s += f"{sep}{_filter:.8f}"
            count += 1
            sep = ", "
            if count % 8 == 0:
                s += ",\n\t"
                sep = ""

            if count % 256 == 0:
                s += "\n\t"

    s += "};\n"

    with open("../csrc/log-mel-filter-bank.h", "w") as f:
        f.write(s)


if __name__ == "__main__":
    main()
