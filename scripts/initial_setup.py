from homr import main as homr_main_file


def main():
    print("This is the initial setup script.")
    print("Downloading weights.")
    # GPU = False für Docker-Container ohne NVIDIA
    homr_main_file.download_weights(use_gpu_inference=False)
    print("Downloading OCR weights.")
    homr_main_file.download_ocr_weights()
    print("All done!")


if __name__ == "__main__":
    main()
