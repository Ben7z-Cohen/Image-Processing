from keras.models import model_from_json
import matplotlib.pyplot as plt
import sol5 as sol
import sol5_utils as sol5_utils

if __name__ == '__main__':
    # Create new deblurring model and saving weights
    noise_model, num_channels = sol.learn_deblurring_model()
    model_json = noise_model.to_json()
    with open("deblurring.json", "w") as json_file:
        json_file.write(model_json)
    noise_model.save_weights("deblurring.h5")
    print("Saved model")
    
    #Loading json and create model
    json_file = open('deblurring.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # Load weights into new model
    loaded_model.load_weights("deblurring.h5")
    print("Loaded model")
    noise_model = loaded_model
    
    #Test our image
    im = sol.read_image("PutTestImagePath", 1)
    bad_im = sol.random_motion_blur(im, [7])
    restored = sol.restore_image(bad_im, noise_model)
    plt.subplot(2, 2, 1)
    plt.imshow(im, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(bad_im, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(restored.reshape(bad_im.shape), cmap='gray')
    plt.show()
    

    #Testing all our pictures
    #Make sure you got the right path in sol5.utils for your images for deblurring
    images = sol5_utils.images_for_deblurring()
    for path in images:
        im = sol.read_image(path, 1)
        bad_im = sol.random_motion_blur(im, [7])
    
        restored = sol.restore_image(bad_im, noise_model)
    
        plt.subplot(2,2,1)
        plt.imshow(im, cmap='gray')
        plt.subplot(2,2,2)
        plt.imshow(bad_im, cmap='gray')
        plt.subplot(2,2,3)
        plt.imshow(restored.reshape(bad_im.shape), cmap='gray')
    
        plt.show()