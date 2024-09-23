import os


def get_image_generation_models_dictionary() -> dict:
    models_dictionary: dict = {}

    current_working_directory = os.getcwd()
    models_folder = f"{current_working_directory}/extensions/stripper/models/image generation models"
    files = os.listdir(models_folder)

    for file in files:
        filename = os.path.basename(file)
        file_path = f"{models_folder}/{filename}"

        models_dictionary[file_path] = file_path

        #full_path = os.path.abspath(file)
        #models_dictionary[filename] = full_path
            
    return models_dictionary
