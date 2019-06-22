from keras.models import model_from_json
import matplotlib.pyplot as plt
import sol5 as sol
import sol5_utils as sol5_utils

if __name__ == '__main__':
    # Create new denoising model and saving weights
    noise_model, num_channels = sol.learn_denoising_model()
    model_json = noise_model.to_json()
    #Create den_json
    with open("denoising.json", "w") as json_file:
        json_file.write(model_json)
    noise_model.save_weights("denoising.h5")
    print("Saved model")

    #Loading Json and Model
    json_file = open('denoising.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("denoising.h5")
    print("Loaded model")
    noise_model = loaded_model
    

    #Showing the result on our test imags
    im = sol.read_image("PutTestImagePath", 1)
    bad_im = sol.add_gaussian_noise(im, 0, 0.2)
    restored = sol.restore_image(bad_im, noise_model)

    plt.subplot(2, 2, 1)
    plt.imshow(im, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(bad_im, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(restored.reshape(bad_im.shape), cmap='gray')
    plt.show()


    #testing all our pictures
    #Make sure you got the right path in sol5.utils for imags to denoising
    images = sol5_utils.images_for_denoising()

    for i in range(len(images)):

        im = sol.read_image(images[i], 1)
        bad_im = sol.add_gaussian_noise(im, 0.2, 0.2)
        restored = sol.restore_image(bad_im, noise_model)

        plt.subplot(2,2,1)
        plt.imshow(im, cmap='gray')
        plt.subplot(2,2,2)
        plt.imshow(bad_im, cmap='gray')
        plt.subplot(2,2,3)
        plt.imshow(restored.reshape(bad_im.shape), cmap='gray')
        plt.show()