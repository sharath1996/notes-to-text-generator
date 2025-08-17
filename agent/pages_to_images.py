from pydantic import BaseModel, Field
import fitz
import base64


class PagesToImagesInput(BaseModel):
    str_inputPath: str = Field(..., description="Path to the input file")
    str_outputPath: str = Field(..., description="Path to the output file")
    bool_saveImages: bool = Field(default=True, description="Flag to save images")

class PagesToImagesOutput(BaseModel):
    list_images: list[str] = Field(..., description="List of image data")

class PagesToImages:


    def __init__(self):
        self._list_images = []
    
    def convert(self, param_obj_input:PagesToImagesInput) -> PagesToImagesOutput:
        """
       
        """
        local_obj_doc = fitz.open(param_obj_input.str_inputPath)
        local_int_pagelength = len(local_obj_doc)
        local_str_pageFormat = self.get_page_number_format(local_int_pagelength)
        for local_obj_page in local_obj_doc:
            local_obj_pix = local_obj_page.get_pixmap()
            if param_obj_input.bool_saveImages:
                local_obj_pix.save(
                f"{param_obj_input.str_outputPath}/page_{local_obj_page.number + 1:{local_str_pageFormat}}.png"
                )
            image_bytes = local_obj_pix.tobytes("png")
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            self._list_images.append(image_base64)
        
        return PagesToImagesOutput(list_str_images=self._list_images)
    
    def get_page_number_format(self, param_int_pages: int) -> str:
            param_int_numberOfDgits = len(str(param_int_pages))
            return f"0{param_int_numberOfDgits}d"
    
