from homr import main as homr_main_file

# Define default configurations
config = homr_main_file.ProcessingConfig(False, False, False, False, -1, False)
xml_generator_args = homr_main_file.XmlGeneratorArguments(False, False, False)


def main():
    print("This is the initial setup script.")
    print("Downloading weights.")
    homr_main_file.download_weights(False)
    print("Downloading OCR weights.")
    homr_main_file.download_ocr_weights()
    print("All done!")


if __name__ == "__main__":
    main()
