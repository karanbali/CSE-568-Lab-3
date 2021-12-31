# Import dependencies
from PIL import Image
import numpy as np
import sys

# Get Filename
filename = sys.argv[1]

# Get Image by filename
im = np.array(Image.open(filename))

# SSD Function ( Also produces "unaligned image")
def ssd(im):

    # Holder for the image manipulation
    new_im = np.array([])
    ssd_rg = np.array([])

    # Height & Width of a single channel image (i.e. R or G or B)
    h = (im.shape[0])/3
    w = (im.shape[1])

    # Get RGB Images from the image (While trimming borders)
    new_r = im[2*h+30:3*h-30,30:w-30]
    new_g = im[h+30:2*h-30,30:w-30]
    new_b = im[30:h-30,30:w-30]

    # Get RGB Images from the image (While trimming borders)
    unaligned_r = im[2*h:3*h,:w]
    unaligned_g = im[h:2*h,:w]
    unaligned_b = im[:h,:w]

    # First part of the assignment (i.e. produce an unaligned_image)
    unaligned_image = np.dstack((unaligned_r,unaligned_g, unaligned_b))

    # Final SSD image
    unaligned_image = Image.fromarray(unaligned_image)
    unaligned_image.show()
    unaligned_image.save("{}-color.jpg".format(filename.split(".")[0]))

    # Initializing minimum values for 'RG' comparison
    min_rg = [0,0]
    min_rg_ssd = np.sum(np.subtract(new_r,new_g, dtype=np.int64)**2)


    # Looping through different permutations of translations (-15 to 15)
    for i in range(-15,15):
        for j in range(-15,15):

            # Translate Green image for given (i,j)
            new_g = im[h+30+i:2*h-30+i,30+j:w-30+j]
        
            # Calculate SSD for RG comparison
            ssd_rg = np.sum(np.subtract(new_r,new_g, dtype=np.int64)**2)
            
            # Check if the SSD is best (i.e. lowest) & update accordignly
            if min_rg_ssd > ssd_rg:
                min_rg_ssd = ssd_rg
                min_rg = [i,j]
            

  

    # Initializing minimum values for 'RB' comparison
    min_rb = [0,0]
    min_rb_ssd = np.sum(np.subtract(new_r,new_b, dtype=np.int64)**2)

    # Looping through different permutations of translations (-15 to 15)
    for i in range(-15,15):
        for j in range(-15,15):

            # Translate Blue image for given (i,j)
            new_b = im[30+i:h-30+i,30+j:w-30+j]

            # Calculate SSD for RB comparison
            ssd_rb = np.sum(np.subtract(new_r,new_b, dtype=np.int64)**2)

            # Check if the SSD is best (i.e. lowest) & update accordignly
            if min_rb_ssd > ssd_rb:
                min_rb_ssd = ssd_rb
                min_rb = [i,j]
            

    # Getting the best translations for 'G' & 'B' channel image 
    rg_x = min_rg[0]
    rg_y = min_rg[1]
    rb_x = min_rb[0]
    rb_y = min_rb[1]

    # Print best SSD Translations
    print("({}) SSD Alignment".format(filename))
    print("Green = [{},{}]".format(rg_x,rg_y))
    print("Blue = [{},{}]".format(rb_x,rb_y))


    # Performing Best translations on 'G' & 'B' channel image ('R' is left untouched)
    new_r = im[2*h+15:3*h-15,15:w-15]
    new_g = im[h+15+rg_x:2*h-15+rg_x,15+rg_y:w-15+rg_y]
    new_b = im[15+rb_x:h-15+rb_x,15+rb_y:w-15+rb_y]

    # Stacking RGB Channel images to get the Final SSD image
    new_im = np.dstack((new_r,new_g, new_b))

    # Showing & saving Final SSD image
    new_im = Image.fromarray(new_im)
    new_im.show()
    new_im.save("{}-ssd.jpg".format(filename.split(".")[0]))

# NCC Function
def ncc(im):

    # Holder for the image manipulation
    new_im = np.array([])
    ncc_rg = np.array([])

    # Height & Width of a single channel image (i.e. R or G or B)
    h = (im.shape[0])/3
    w = (im.shape[1])

    # Get RGB Images from the image (While trimming borders)
    new_r = im[2*h+15:3*h-15,15:w-15]
    new_g = im[h+15:2*h-15,15:w-15]
    new_b = im[15:h-15,15:w-15]

    # Getting 'mean' for each of the RGB channel image
    norm_r = new_r.mean(dtype=np.int64)
    norm_g = new_g.mean(dtype=np.int64)
    norm_b = new_b.mean(dtype=np.int64)

    # Subtracting 'mean' from each of the RGB channel image respectively
    new_r = np.subtract(new_r, norm_r, dtype=np.int64)
    new_g = np.subtract(new_g, norm_g, dtype=np.int64)
    new_b = np.subtract(new_b, norm_b, dtype=np.int64)

    # Calculate 'norm' for each of the RGB channel image
    norm_r = np.linalg.norm(new_r)
    norm_g = np.linalg.norm(new_g)
    norm_b = np.linalg.norm(new_b)


    # Calculate 'normalized image' for each of the RGB channel image
    new_r = new_r / norm_r
    new_g = new_g / norm_g
    new_b = new_b / norm_b

    
    # Initializing 'R' & 'G' translated channel images for the loop ahead
    new_gt = new_g[0:(-30+0),0:(-30+0)]
    new_rt = new_r[0:(-30+0),0:(-30+0)]

    # Initializing best values for 'RG' comparison
    min_rg = [0,0]
    min_rg_ncc = np.sum(np.multiply(new_rt,new_gt))



    # Looping through different permutations of translations (-15 to 15)
    for i in range(30):
        for j in range(30):
            
            # Translate Green image for given (i,j)
            new_gt = new_g[i:(-30+i),j:(-30+j)]
            
            # Calculate NCC for RG comparison
            ncc_rg = np.sum(np.multiply(new_rt,new_gt))
            
            # Check if the NCC is best (i.e. highest) & update accordignly
            if min_rg_ncc < ncc_rg:
                min_rg_ncc = ncc_rg
                min_rg = [i,j]
       

    # Initializing 'R' & 'B' translated channel images for the loop ahead
    new_bt = new_b[0:(-30+0),0:(-30+0)]
    new_rt = new_r[0:(-30+0),0:(-30+0)]

    # Initializing best values for 'RB' comparison
    min_rb = [0,0]
    min_rb_ncc = np.sum(new_rt*new_bt)


    # Looping through different permutations of translations (-15 to 15)
    for i in range(30):
        for j in range(30):
            
            # Translate Blue image for given (i,j)
            new_bt = new_b[i:(-30+i),j:(-30+j)]
        
            # Calculate NCC for RB comparison
            ncc_rb = np.sum(new_rt*new_bt)
            
            # Check if the NCC is best (i.e. highest) & update accordignly
            if min_rb_ncc < ncc_rb:
                min_rb_ncc = ncc_rb
                min_rb = [i,j]
    

    # Getting the best translations for 'G' & 'B' channel image
    rg_x = min_rg[0]
    rg_y = min_rg[1]
    rb_x = min_rb[0]
    rb_y = min_rb[1]


    # Print best NCC translations
    print("({}) NCC Alignment".format(filename))
    print("Green = [{},{}]".format(rg_x,rg_y))
    print("Blue = [{},{}]".format(rb_x,rb_y))
    

    # Performing Best translations on 'G' & 'B' channel image ('R' is left untouched)
    new_r = im[2*h:3*h-30,:w-30]
    new_g = im[h+rg_x:2*h+ (-30+rg_x),rg_y:w+(-30+rg_y)]
    new_b = im[rb_x:h + (-30+rb_x),rb_y:w+(-30+rb_y)]

    
    # Stacking RGB Channel images to get the Final NCC image
    new_im = np.dstack((new_r,new_g, new_b))

    # Saving & Showing Final NCC image
    new_im = Image.fromarray(new_im)
    new_im.show()  
    new_im.save("{}-ncc.jpg".format(filename.split(".")[0]))

# Calling SSD & NCC Function
ssd(im)
ncc(im)

# Note : 'ssd(im)' also produces 'unaligned image'