from PIL import Image
import face_recognition
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_image', type = str, default = './sample_photo.jpeg')
parser.add_argument('--cropped_image_dir', type = str, default = './cropped')

if __name__ == '__main__':
    
    args = parser.parse_args()
    
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(args.input_image)

    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
    face_locations = face_recognition.face_locations(image)

    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    for i, face_location in enumerate(face_locations):

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        #pil_image.show()

        save_path = os.path.join(args.cropped_image_dir, 'face{}.jpeg'.format(i))
        pil_image.save(save_path)
