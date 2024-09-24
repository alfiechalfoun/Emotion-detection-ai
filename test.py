
import cv2 
from PIL import Image
import pillow_heif
import numpy as np 

#could use to add more imiges into training 
photo = '/Users/alfie/Desktop/pearce.HEIC'
class Maxpooling_padded():
    
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

    def pad_imiage (self, im):
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
            except:
                self.channels = 1
            
            new_height = (self.height + 47) // 48 * 48 
            new_width = (self.width + 47) // 48 * 48

            pad_top = (new_height - self.height) // 2
            pad_bottom = new_height - self.height - pad_top
            pad_left = (new_width - self.width) // 2
            pad_right = new_width - self.width - pad_left

            self.padded_image = cv2.copyMakeBorder(
                self.image, 
                pad_top, pad_bottom, pad_left, pad_right, 
                cv2.BORDER_CONSTANT, 
                value=[0, 0, 0]  # Padding with black pixels
            )

            return self.padded_image



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
        Croped_image = self.pad_imiage (im)

        # Now apply max pooling
        # Assuming the image is padded to multiples of 48, apply max pooling with a pool size
        # This will reduce the image dimensions to 48x48
        pool_size = (Croped_image.shape[0] // 48, Croped_image.shape[1] // 48)
        self.pooled_image = self.Maxpool(Croped_image, pool_size)
        
        return self.pooled_image

if __name__ == '__main__':
    obj = Maxpooling_padded()
    obj.resize_with_max_pooling(photo)