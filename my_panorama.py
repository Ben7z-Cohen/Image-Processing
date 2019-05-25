import os
import sol4
import time

TRANSLATION_ONLY=False
def main():
    # video from my trip to south-america
    experiment = 'Bariloche_argentina_for_panorama.mp4'



    exp_no_ext = experiment.split('.')[0]
    os.system('mkdir dump')
    os.system('mkdir dump/%s' % exp_no_ext)
    os.system('ffmpeg -i videos/%s dump/%s/%s%%03d.jpg' % (experiment, exp_no_ext, exp_no_ext))

    s = time.time()
    panorama_generator = sol4.PanoramicVideoGenerator('dump/%s/' % exp_no_ext, exp_no_ext, 2100)
    panorama_generator.align_images(translation_only=TRANSLATION_ONLY)
    panorama_generator.generate_panoramic_images(15)
    print(' time for %s: %.1f' % (exp_no_ext, time.time() - s))
    panorama_generator.save_panoramas_to_video()


if __name__ == '__main__':
  main()
