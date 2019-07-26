## If you find this code useful, please provide a reference to my github page for others www.github.com/brianmanderson , thank you!
# Create_RT_From_Dicom_and_Mask
This is for the creation of a Dicom_RT structure using a template RT structure, the dicom images, and a mask

Conversion of dicom images + given annotations into an RT structure
    

Example:

    from Create_RT_From_Dicom_and_mask import main

    image_path = 'C:\users\brianmanderson\Patient_1\CT1\'

    Contour_Names = ['Liver']
    annotations = numpy_array_predictions
    
    # annotations should have the shape of [# images, 512, 512, #classes + 1]
    # annotations[...,0] is the background
    
    out_path = 'C:\users\brianmanderson\Patient_1\Output\'
    main(image_path=image_path,annotations=annotations,Contour_Names=Contour_Names,out_path=out_path)
