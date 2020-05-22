import restore_corrupted_image as cor

if __name__ == '__main__':

    corrupted_object_choice = int(input("Please choose which model of corruption to use:\n"
                                        "1.Deblurring\n"
                                        "2.Denoising\n"))
    try:
        corrupted_object = cor.RestoreCorruptedImage.factory(corrupted_object_choice)
        task = int(input("Please choose how to proceed:\n"
                         "1.Train model.\n"
                         "2.Test one image.\n"
                         "3.Test set of images.\n"))
        if task == 1:
            corrupted_object.save_trained_model()
            print("The model was trained, to test the results please run again.")
        elif task == 2:
            path = input("Please enter the path of the image you wish to test.\n")
            corrupted_object.test_image(path)
        elif task == 3:
            corrupted_object.test_images()
        else:
            raise Exception("Wrong number was entered, please run again.")

    except Exception as error:
        print("Error: " + str(error))



