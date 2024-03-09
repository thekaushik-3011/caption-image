from langchain.tools import BaseTool
from function import get_image_caption, detect_objects

class ImageCaptionTool(BaseTool):
    name = "Image Captioner"
    description = "Use this tool when given the path to an image that you would like to be described."\
                  "It will return a simple caption describing the image."

    def _run(self, img_path):
        get_image_caption(img_path)
    
    def _async_run(self, query:str):
        raise NotImplementedError("This tool does not support async")
    

class ObjectDetectionTool(BaseTool):
    name = "Object Detector"
    description = "Use this tool when given the path to an image taht you would like to detect objects."\
                  "It will return a list of all detetcted objects. Each element in the list in the format: "\
                  "[x1, y1, x2, y2] class_name confidence_score."  

    def _run(self, img_path):
        detect_objects(img_path)

    def _async_run(self, query:str):
        raise NotImplementedError("This tool does not support async")
