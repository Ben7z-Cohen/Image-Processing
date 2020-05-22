from enum import Enum
from abc import abstractmethod
import sol5 as sol
import sol5_utils as sol5_utils
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import pathlib
from os import path

class RestoreCorruptedType(Enum):
    Deblurring = 1,
    Denoising = 2,


class RestoreCorruptedImage:

    LOAD_MODEL_ERROR = "Error: Please choose option 1 to create model files"

    @abstractmethod
    def save_trained_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def test_image(self, image_path):
        pass

    @abstractmethod
    def test_images(self):
        pass

    @staticmethod
    def factory(corrupted_type):
        if corrupted_type == RestoreCorruptedType.Deblurring.value[0]:
            return Deblurring()
        if corrupted_type == RestoreCorruptedType.Denoising.value[0]:
            return Denoising()
        raise ValueError("Error: Bad corrupted model creation: " + RestoreCorruptedType(corrupted_type).name)

    @property
    def json_file(self):
        return self._json_file

    @json_file.setter
    def json_file(self, value):
        self._json_file = value

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    @property
    def trained_files_exists(self):
        return path.exists(self.current_path() + "\\" + self.json_file) \
               and path.exists(self.current_path() + "\\" + self.weights)

    @staticmethod
    def current_path():
        return str(pathlib.Path(__file__).parent.absolute())

class RestoreCorruptedHandler:

    def save_trained_model(self, model, json_file, weights, current_path):
        try:
            model = model()
        except:
            raise Exception("Error: Please check if you've downloaded the images for training to the "
                            "image dataset/train folder.\nrun again.")

        model_json = model.to_json()
        with open(current_path + "\\" + json_file, "w+") as json_file:
            json_file.write(model_json)
        model.save_weights(current_path + "\\" + weights)
        print("Saved model")

    def load_model(self, json_name, weights, current_path):
        with open(current_path + "\\" + json_name, "r") as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(current_path + "\\" + weights)
        print("Loaded model")
        return loaded_model

    def test_image(self, loaded_model, image_path, random_corrupt, random_corrupt_factor):
        try:
            image = sol.read_image(image_path, sol.Representation.RGB)
        except:
            raise Exception("Wrong image path was received, "
                            "please put the correct path for the tested image.\run again")

        random_corrupt_factor = [random_corrupt_factor] if random_corrupt_factor > 1 else random_corrupt_factor
        bad_image = random_corrupt(image, random_corrupt_factor)
        restored = sol.restore_image(bad_image, loaded_model)
        plt.subplot(2, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.subplot(2, 2, 2)
        plt.imshow(bad_image, cmap='gray')
        plt.subplot(2, 2, 3)
        plt.imshow(restored.reshape(bad_image.shape), cmap='gray')
        plt.show()

    def test_images(self, loaded_model, images, random_corrupt, list_of_kernel_sizes):
        try:
            for path in images:
                self.test_image(loaded_model, path, random_corrupt, list_of_kernel_sizes)
        except:
            raise Exception(" Wrong paths for the images was received, "
                             "please check if you've downloaded the provided images for testing to"
                             "image_dataset/train folder.\n"
                             "run again")


class Deblurring(RestoreCorruptedImage):

    def __init__(self, json_file="deblurring.json", weights="deblurring.h5"):
        self._corrupted_handler = RestoreCorruptedHandler()
        self._json_file = json_file
        self._weights = weights
        self._random_corrupt = lambda x, y: sol.random_motion_blur(x, y)
        self._list_of_kernel_sizes = 7
        self._model = sol.learn_deblurring_model
        self._corrupted_images = sol5_utils.images_for_deblurring()

    def save_trained_model(self):
        self._corrupted_handler.save_trained_model(self._model,
                                                   self._json_file, self._weights, self.current_path())

    def load_model(self):
        if self.trained_files_exists:
            return self._corrupted_handler.load_model(self._json_file, self._weights, self.current_path())
        raise Exception(self.LOAD_MODEL_ERROR)

    def test_image(self, image_path):
        self._corrupted_handler.test_image(self.load_model(), image_path, self._random_corrupt,
                                           self._list_of_kernel_sizes)

    def test_images(self):
        self._corrupted_handler.test_images(self.load_model(), self._corrupted_images,
                                            self._random_corrupt, self._list_of_kernel_sizes)


class Denoising(RestoreCorruptedImage):

    MAX_SIGMA = 0.2

    def __init__(self, json_file="denoising.json", weights="denoising.h5"):
        self._corrupted_handler = RestoreCorruptedHandler()
        self._json_file = json_file
        self._weights = weights
        self._random_corrupt = lambda x, y: sol.add_gaussian_noise(x, y, self.MAX_SIGMA)
        self._model = sol.learn_denoising_model
        self._corrupted_images = sol5_utils.images_for_denoising()

    def save_trained_model(self):
        self._corrupted_handler.save_trained_model(self._model,
                                                   self._json_file, self._weights, self.current_path())

    def load_model(self):
        if self.trained_files_exists:
            return self._corrupted_handler.load_model(self._json_file, self._weights, self.current_path())
        raise Exception(self.LOAD_MODEL_ERROR)

    def test_image(self, image_path):
        min_sigma = 0
        self._corrupted_handler.test_image(self.load_model(), image_path, self._random_corrupt,
                                           min_sigma)

    def test_images(self):
        min_sigma = 0.2
        self._corrupted_handler.test_images(self.load_model(), self._corrupted_images,
                                            self._random_corrupt, min_sigma)
