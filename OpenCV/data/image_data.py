import os
import skimage
import skimage.io
import skimage.transform


def prepare_image(folder_dir='D:\image'):
    image_list = []
    find_image(folder_dir, image_list)
    return image_list


def find_image(file_dir, file_list):
    results = [os.path.join(file_dir, result) for result in os.listdir(file_dir)]
    for result in results:
        if os.path.isdir(result):
            find_image(result, file_list)
        else:
            file_list.append(result)
    pass


def save_image(image, image_dir, image_number):
    base_dir = '../images'

    (file_path, complete_filename) = os.path.split(image_dir)
    (filename, extension) = os.path.splitext(complete_filename)
    new_filename = str(image_number) + str(extension)

    data_dir = os.path.join(base_dir, 'data', new_filename)
    print('save' + data_dir)
    label_dir = os.path.join(base_dir, 'label', new_filename)
    print('save' + label_dir)

    image_resize = resize_image(image, label_dir, edge=512)
    resize_data(image_resize, data_dir)
    pass


def resize_data(image,  image_dir):
    image_resize = skimage.transform.resize(image, (128, 128), mode='constant')
    new_image_resize = skimage.transform.resize(image_resize, (512, 512), mode='constant')
    skimage.io.imsave(image_dir, new_image_resize)
    pass


def resize_image(image,  image_dir, edge=512):
    short_edge = min(image.shape[:2])
    start_x = int((image.shape[1] - short_edge) / 2)
    start_y = int((image.shape[0] - short_edge) / 2)
    crop_image = image[start_y: start_y + short_edge, start_x: start_x + short_edge]
    image_resize = skimage.transform.resize(crop_image, (edge, edge), mode='constant')
    skimage.io.imsave(image_dir, image_resize)
    return image_resize
    pass


def convert_image(image_list):
    image_number = 0
    for image_dir in image_list:
        try:
            image = skimage.io.imread(image_dir)
            image_number += 1
            save_image(image, image_dir, image_number)
        except:
            print(image_dir + " not an image")
    pass


def main():
    image_list = prepare_image()
    convert_image(image_list)
    pass


if __name__ == '__main__':
    main()
