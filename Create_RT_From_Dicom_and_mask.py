__author__ = 'Brian Mark Anderson'
__email__ = 'bmanderson@mdanderson.org'

import pydicom, os, copy
import numpy as np
from skimage import draw
from skimage.measure import label,regionprops,find_contours

class Make_RT_Data:
    def __init__(self,delete_previous_rois=True):
        self.delete_previous_rois = delete_previous_rois

    def get_images(self,dir_to_dicom, single_structure=True):
        self.single_structure = single_structure
        self.dir_to_dicom = dir_to_dicom
        self.lstFilesDCM = []
        self.Dicom_info = {}
        self.lstRSFile = []
        i = 0
        fileList = []
        for dirName, _, fileList in os.walk(self.dir_to_dicom):
            break
        for filename in fileList:
            print(str(fileList.index(filename)/len(fileList) * 100) + '% Loading')
            i += 1
            try:
                ds = pydicom.read_file(os.path.join(dirName,filename))
                if ds.Modality != 'RTSTRUCT':  # check whether the file's DICOM
                    self.lstFilesDCM.append(os.path.join(dirName, filename))
                    self.Dicom_info[os.path.join(dirName, filename)] = ds
                    self.SeriesInstanceUID = ds.SeriesInstanceUID
                elif ".dcm" in filename.lower() and ds.Modality == 'RTSTRUCT':
                    self.lstRSFile = os.path.join(dirName, filename)
            except:
                continue
        self.num_images = len(self.lstFilesDCM)
        self.get_images_and_mask()

    def get_images_and_mask(self):
        self.slice_info = np.zeros([len(self.lstFilesDCM)])
        # Working on the RS structure now
        self.template = True
        if self.template:
            self.template_dir = os.path.join('\\\\mymdafiles','ro-admin','SHARED','Radiation physics','BMAnderson','Auto_Contour_Sites','template_RS.dcm')
            self.template_dir = os.path.join('.','template_RS.dcm')
            self.key_list = self.template_dir.replace('template_RS.dcm','key_list.txt')
            if not os.path.exists(self.template_dir):
                self.template_dir = os.path.join('..','..','Shared_Drive','Auto_Contour_Sites','template_RS.dcm')
                self.key_list = self.template_dir.replace('template_RS.dcm','key_list.txt')
            self.RS_struct = pydicom.read_file(self.template_dir)

        # Get ref file
        self.RefDs = pydicom.read_file(self.lstFilesDCM[0])

        ds = self.Dicom_info[self.lstFilesDCM[0]].pixel_array
        self.ArrayDicom = np.zeros([self.num_images,ds.shape[0], ds.shape[1], 1], dtype='float32')
        self.SOPClassUID_temp =[None] * self.num_images
        self.SOPClassUID = [None] * self.num_images
        for filenameDCM in self.lstFilesDCM:
            ds = self.Dicom_info[filenameDCM]
            im = ds.pixel_array
            im = np.array([im for i in range(1)]).transpose([1,2,0])
            self.ArrayDicom[self.lstFilesDCM.index(filenameDCM),:, :, :] = im
            self.slice_info[self.lstFilesDCM.index(filenameDCM)] = round(ds.ImagePositionPatient[2],3)
            self.SOPClassUID_temp[self.lstFilesDCM.index(filenameDCM)] = ds.SOPInstanceUID
        indexes = [i[0] for i in sorted(enumerate(self.slice_info),key=lambda x:x[1])]
        self.ArrayDicom = self.ArrayDicom[indexes]
        self.slice_info = self.slice_info[indexes]
        try:
            self.ArrayDicom = (self.ArrayDicom + ds.RescaleIntercept) / ds.RescaleSlope
        except:
            xxx = 1
        i = 0
        for index in indexes:
            self.SOPClassUID[i] = self.SOPClassUID_temp[index]
            i += 1
        self.ds = ds
        if self.template:
            print('Running off a template')
            self.changetemplate()

    def with_annotations(self,annotations,ROI_Names=None):
        '''
        :param annotations: Numpy array of images in the form of #_images, 512, 512, #_classes + 1 (0 is background)
        :param ROI_Names: A list of ROI names ['Roi - 1']
        :return:
        '''
        annotations = np.squeeze(annotations)
        if len(annotations.shape) == 3:
            annotations = np.expand_dims(annotations,axis=-1)
        self.image_size_0, self.image_size_1 = annotations.shape[1], annotations.shape[2]
        self.ROI_Names = ROI_Names
        self.annotations = annotations
        self.Mask_to_Contours()
    def Mask_to_Contours(self):
        self.RefDs = self.ds
        self.ShiftCols = self.RefDs.ImagePositionPatient[0]
        self.ShiftRows = self.RefDs.ImagePositionPatient[1]
        self.mult1 = self.mult2 = 1
        self.PixelSize = self.RefDs.PixelSpacing[0]
        current_names = []
        for names in self.RS_struct.StructureSetROISequence:
            current_names.append(names.ROIName)
        Contour_Key = {}
        xxx = 1
        for name in self.ROI_Names:
            Contour_Key[name] = xxx
            xxx += 1
        self.all_annotations = self.annotations
        base_annotations = copy.deepcopy(self.annotations)
        temp_color_list = []
        color_list = [[128,0,0],[170,110,40],[0,128,128],[0,0,128],[230,25,75],[225,225,25],[0,130,200],[145,30,180],
                      [255,255,255]]
        for Name in self.ROI_Names:
            if not temp_color_list:
                temp_color_list = copy.deepcopy(color_list)
            color_int = np.random.randint(len(temp_color_list))
            print('Writing data for ' + Name)
            self.annotations = base_annotations[:,:,:,int(self.ROI_Names.index(Name)+1)]
            self.annotations = self.annotations.astype('int')

            make_new = 1
            allow_slip_in = True
            if Name not in current_names and allow_slip_in:
                self.RS_struct.StructureSetROISequence.append(copy.deepcopy(self.RS_struct.StructureSetROISequence[0]))
                if not self.template:
                    self.struct_index = len(self.RS_struct.StructureSetROISequence)-1
                else:
                    self.struct_index = 0
            else:
                make_new = 0
                self.struct_index = current_names.index(Name) - 1
            new_ROINumber = self.struct_index + 1
            self.RS_struct.StructureSetROISequence[self.struct_index].ROINumber = new_ROINumber
            self.RS_struct.StructureSetROISequence[self.struct_index].ReferencedFrameOfReferenceUID = self.ds.FrameOfReferenceUID
            self.RS_struct.StructureSetROISequence[self.struct_index].ROIName = Name
            self.RS_struct.StructureSetROISequence[self.struct_index].ROIVolume = 0
            self.RS_struct.StructureSetROISequence[self.struct_index].ROIGenerationAlgorithm = 'SEMIAUTOMATIC'
            if make_new == 1:
                self.RS_struct.RTROIObservationsSequence.append(copy.deepcopy(self.RS_struct.RTROIObservationsSequence[0]))
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ObservationNumber = new_ROINumber
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ReferencedROINumber = new_ROINumber
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ROIObservationLabel = Name
            self.RS_struct.RTROIObservationsSequence[self.struct_index].RTROIInterpretedType = 'ORGAN'

            if make_new == 1:
                self.RS_struct.ROIContourSequence.append(copy.deepcopy(self.RS_struct.ROIContourSequence[0]))
            self.RS_struct.ROIContourSequence[self.struct_index].ReferencedROINumber = new_ROINumber
            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[1:] = []
            self.RS_struct.ROIContourSequence[self.struct_index].ROIDisplayColor = temp_color_list[color_int]
            del temp_color_list[color_int]

            contour_num = 0
            if np.max(self.annotations) > 0: # If we have an annotation, write it
                image_locations = np.max(self.annotations,axis=(1,2))
                indexes = np.where(image_locations>0)[0]
                for point, i in enumerate(indexes):
                    print(str(int(point / len(indexes) * 100)) + '% done with ' + Name)
                    annotation = self.annotations[i,:,:]
                    regions = regionprops(label(annotation),coordinates='xy')
                    for ii in range(len(regions)):
                        temp_image = np.zeros([self.image_size_0,self.image_size_1])
                        data = regions[ii].coords
                        rows = []
                        cols = []
                        for iii in range(len(data)):
                            rows.append(data[iii][0])
                            cols.append(data[iii][1])
                        temp_image[rows,cols] = 1
                        points = find_contours(temp_image, 0)[0]
                        output = []
                        for point in points:
                            output.append(((point[1]) * self.PixelSize + self.mult1 * self.ShiftCols))
                            output.append(((point[0]) * self.PixelSize + self.mult2 * self.ShiftRows))
                            output.append(self.slice_info[i])
                        if output:
                            if contour_num > 0:
                                self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence.append(copy.deepcopy(self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[0]))
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[contour_num].ContourNumber = str(contour_num)
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[contour_num].ContourImageSequence[0].ReferencedSOPInstanceUID = self.SOPClassUID[i]
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[contour_num].ContourData = output
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[contour_num].NumberofContourPoints = round(len(output)/3)
                            contour_num += 1
        self.RS_struct.SOPInstanceUID += '.' + str(np.random.randint(999))
        if self.template and self.delete_previous_rois:
            for i in range(len(self.RS_struct.StructureSetROISequence) - len(self.ROI_Names), -1 + len(self.ROI_Names), -1):
                del self.RS_struct.StructureSetROISequence[i]
            for i in range(len(self.RS_struct.RTROIObservationsSequence) - len(self.ROI_Names), -1 + len(self.ROI_Names),
                           -1):
                del self.RS_struct.RTROIObservationsSequence[i]
            for i in range(len(self.RS_struct.ROIContourSequence) - len(self.ROI_Names), -1 + len(self.ROI_Names),
                           -1):
                del self.RS_struct.ROIContourSequence[i]
        for i in range(len(self.RS_struct.StructureSetROISequence)):
            self.RS_struct.StructureSetROISequence[i].ROINumber = i + 1
            self.RS_struct.RTROIObservationsSequence[i].ReferencedROINumber = i + 1
            self.RS_struct.ROIContourSequence[i].ReferencedROINumber = i + 1

        return None

    def changetemplate(self):
        keys = self.RS_struct.keys()
        for key in keys:
            #print(self.RS_struct[key].name)
            if self.RS_struct[key].name == 'Referenced Frame of Reference Sequence':
                break
        self.RS_struct[key]._value[0].FrameOfReferenceUID = self.ds.FrameOfReferenceUID
        self.RS_struct[key]._value[0].RTReferencedStudySequence[0].ReferencedSOPInstanceUID = self.ds.StudyInstanceUID
        self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID = self.ds.SeriesInstanceUID
        for i in range(len(self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence)-1):
            del self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence[0]
        for i in range(len(self.SOPClassUID)):
            self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence[i].ReferencedSOPInstanceUID = self.SOPClassUID[i]
            self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence.append(copy.deepcopy(self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence[0]))
        del self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence[-1]

        things_to_change = ['StudyInstanceUID','Specific Character Set','Instance Creation Date','Instance Creation Time','Study Date','Study Time',
                            'Accession Number','Study Description','Patient"s Name','Patient ID','Patients Birth Date','Patients Sex'
                            'Study Instance UID','Study ID','Frame of Reference UID']
        self.RS_struct.PatientName = self.ds.PatientName
        self.RS_struct.PatientSex = self.ds.PatientSex
        self.RS_struct.PatientBirthDate = self.ds.PatientBirthDate
        for key in keys:
            #print(self.RS_struct[key].name)
            if self.RS_struct[key].name in things_to_change:
                try:
                    self.RS_struct[key] = self.ds[key]
                except:
                    continue
        new_keys = open(self.key_list)
        keys = {}
        i = 0
        for line in new_keys:
            keys[i] = line.strip('\n').split(',')
            i += 1
        new_keys.close()
        for index in keys.keys():
            new_key = keys[index]
            try:
                self.RS_struct[new_key[0], new_key[1]] = self.ds[[new_key[0], new_key[1]]]
            except:
                continue
        return None
            # Get slice locations

    def write_RT_Structure(self, out_dir):
        out_name = os.path.join('RS_MRN' + self.RS_struct.PatientID + '_' + self.RS_struct.SeriesInstanceUID + '.dcm')
        out_path = os.path.join(out_dir,out_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print('writing out RT structure at ' + out_path)
        pydicom.write_file(out_path, self.RS_struct)
        return None

def poly2mask(vertex_row_coords, vertex_col_coords):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, [512,512])
    mask = np.zeros([512,512], dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def main(image_path='',annotations=None,Contour_Names=['Vasculature'],out_path=''):
    '''
    :param image_path: Path to the DICOM images
    :param annotations: Your annotations, in the format of [# Images, rows, cols, classes + 1]
    The class 0 of annotations should be background
    :param Contour_Names: A list of contour names, should be equal to annotations.shape[-1] -1
    :param out_path: Path to output the RT structure
    :return:
    '''
    Dicom_Image_Class = Make_RT_Data()
    Dicom_Image_Class.get_images(image_path)
    Dicom_Image_Class.with_annotations(annotations,Contour_Names)
    Dicom_Image_Class.write_RT_Structure(out_path)
if __name__ == '__main__':
    xxx = 1