# Lucky13
## Handwriting Recognition - Dead Sea Scrolls

This is the main project file for our project

In addition to these files, two trained models are also needed in the same folder:
1) https://drive.google.com/file/d/10NYOP5pGiOxucL2_Muek12nkHeaoL5GJ/view?usp=sharing
2) https://drive.google.com/file/d/1-iW-Ae8dYymQSYifb3BCqeOVLtF4-KBq/view?usp=sharing

- Once all files are in the same folder, the following command can be used to run the program:
```sh
py Segmentation.py path/to/test/images/
```



## Result

- Result folder is created which includes the text files for recognition and classification
- Segmented_Lines folder is created which has cropped lines per image for character recognition
- Segmented_Characters_Per_Lines folder is created to save the cropped characters per line per image.
