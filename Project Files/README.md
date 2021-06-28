# Lucky13
## Handwriting Recognition - Dead Sea Scrolls

This is the main project file for our project

In addition to these files, two trained models are also needed in the same folder:
1) https://drive.google.com/file/d/10NYOP5pGiOxucL2_Muek12nkHeaoL5GJ/view?usp=sharing
2) https://drive.google.com/file/d/1-iW-Ae8dYymQSYifb3BCqeOVLtF4-KBq/view?usp=sharing

- Once all files are in the same folder, the following command can be used to run the program:
```sh
py Segmentation.py images/
```
This command works if the path is to the folder with the images and not to the image directly.

> Warning - If there is space " " in path, commands will not work.


## Files

- Segmentation.py is the main script that runs the program
- Both model files are the trained models for Character and dialect Recognition



## Result

- Result folder is created which includes the text files for recognition and classification
- Segmented_Lines folder is created which has cropped lines per image for character recognition
- Segmented_Characters_Per_Lines folder is created to save the cropped characters per line per image.
