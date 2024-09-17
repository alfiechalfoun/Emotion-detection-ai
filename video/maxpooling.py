import cv2 
from PIL import Image
import pillow_heif
import numpy as np 

#could use to add more imiges into training 
photo = '/Users/alfie/Desktop/pearce.HEIC'
class Maxpooling():
    
    def readimage(self,im):
        if self.im.lower().endswith(".heic"):

            heif_file = pillow_heif.read_heif(self.im)
            image = Image.frombytes(
                heif_file.mode, 
                heif_file.size, 
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride
            )
                # Convert to OpenCV format
            self.image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            # For other formats like JPG/PNG
            self.image = cv2.imread(self.im)

    def crop_imige(self, im):
        self.im = im

        if isinstance(self.im, str):
            self.readimage(im)
        else:
            self.image = self.im

        dimensions = self.image.shape
        self.height = dimensions[0]
        self.width = dimensions[1]

        try:
            self.channels = dimensions[2]
        except IndexError:
            self.channels = 1

        # Calculate the new dimensions to crop to the nearest multiple of 48
        new_height = (self.height // 48) * 48
        new_width = (self.width // 48) * 48

        # Calculate the cropping margins
        crop_top = (self.height - new_height) // 2
        crop_bottom = self.height - new_height - crop_top
        crop_left = (self.width - new_width) // 2
        crop_right = self.width - new_width - crop_left

        # Crop the image
        self.cropped_image = self.image[crop_top:self.height - crop_bottom, crop_left:self.width - crop_right]

        return self.cropped_image

    def Maxpool(self, image, pool_size):
        # Get the shape of the image
        if self.channels == 1:
            height, width = self.image.shape
            channels = 1
        else:
            height, width, channels = self.image.shape

        # Calculate the size of the pooled image
        pooled_height = height // pool_size[0]
        pooled_width = width // pool_size[1]
        pooled_image = np.zeros((pooled_height, pooled_width, channels), dtype=image.dtype)

        # Perform max pooling
        for h in range(pooled_height):
            for w in range(pooled_width):
                for c in range(channels):
                    h_start = h * pool_size[0]
                    w_start = w * pool_size[1]
                    h_end = min(h_start + pool_size[0], height)
                    w_end = min(w_start + pool_size[1], width)

                    # Get the pooling window
                    pooling_window = image[h_start:h_end, w_start:w_end, c] if channels > 1 else image[h_start:h_end, w_start:w_end]

                    # Skip if the pooling window is empty (zero-size)
                    if pooling_window.size == 0:
                        continue

                    # Perform max pooling on the valid window
                    if channels > 1:
                        pooled_image[h, w, c] = np.max(pooling_window)
                    else:
                        pooled_image[h, w] = np.max(pooling_window)

        return pooled_image

    def resize_with_max_pooling(self,im):
        # First pad the image to the nearest multiple of 48
        Croped_image = self.crop_imige(im)

        # Now apply max pooling
        # Assuming the image is padded to multiples of 48, apply max pooling with a pool size
        # This will reduce the image dimensions to 48x48
        pool_size = (Croped_image.shape[0] // 48, Croped_image.shape[1] // 48)
        self.pooled_image = self.Maxpool(Croped_image, pool_size)
        
        return self.pooled_image

if __name__ == '__main__':
    obj = Maxpooling()
    obj.resize_with_max_pooling(photo)