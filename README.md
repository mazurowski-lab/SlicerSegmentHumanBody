# SlicerSegmentAny3D

[![arXiv Paper](https://img.shields.io/badge/arXiv-2401.12974-orange.svg?style=flat)](https://arxiv.org/abs/2401.12974)

SegmentAny3D aims to asist its users in segmenting medical data on <a href="https://github.com/Slicer/Slicer">3D Slicer</a> by integrating the <a href="https://github.com/mazurowski-lab/SegmentAnyBone">SegmentAnyBone</a> developed by Mazurowski Lab.

## License

The repository is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

<img src="SegmentAny3D/Resources/Icons/SegmentAny3D.png" width=50% height=50%>

## Installation via GitHub Repository

You can clone this repository by running the following command:

```
git clone https://github.com/mazurowski-lab/SlicerSegmentAny3D.git
```

After cloning the repository, you need to introduce the extension to 3D Slicer. Please go to Modules > Developer Tools > Extension Wizard on 3D Slicer and click 'Select Extension' button. You should select the root folder that contains this repository in the pop-up. If you don't get any error on Python terminal, that means you are ready to use the extension!

## Usage

If you've added the extension to 3D Slicer, you should be able to see it under **Modules > Segmentation > SegmentAny3D**. You can see the user interface of the extension after you click on SegmentAny3D in this menu.

Before starting the segmentation, make sure that you've created the necessary labels for your case by clicking "Configure labels in the segment editor" button. You need to turn back to our extension through Modules > Segmentation > SegmentAny3D path again, after you create your labels in the segment editor. You are ready to segment now!

### Automatic Segmentation

<img src="Screenshots/sws1.png" width=45%> <img src="Screenshots/sws2.png" width=42%>

If you click "Run Automatic Segmentation" button, SegmentAnyBone will run in automated mode and will segment bones in each slice it can detect. You can assign different labels after you get your binary segmentation. There is 2 different modes for label assignment:

**2D Label Assignment:** This mode is used for changing the label of the connected component only in the current slice. You should first choose your target label and then click "Assign Label (2D)" button.

**3D Label Assignment:** This mode is used for changing the label of the connected component through consecutive slices in 3D space. You should first choose your target label and then click "Assign Label (3D)" button.

### Prompt Based Segmentation

<img src="Screenshots/sws3.png" width=45% height=45%>

Firstly, select the label you want to segment from the dropdown list (hip for the image below). Then, click "Start Segmentation for Current Slice" button.

If it is the first to segment a slice of this file, you need to wait for SegmentAnyBone to produce some files that will be used for the segmentation. After SegmentAnyBone generated these files, you can start putting **prompt points** or **prompt boxes** on the current slice. You'll be able to see the segmentation mask on 3D Slicer. Please click "Stop Segmentation for Current Slice" whenever you finish your segmentation for the current slice.

If you are not satisfied with the segmentation mask produced by SegmentAnyBone, you can edit it as you wish using the "Segment Editor" module of 3D Slicer.


## Citation

If you find our work to be useful for your research, please cite [our paper](https://arxiv.org/abs/2401.12974):

```bibtex
@misc{gu2024segmentanybone,
      title={SegmentAnyBone: A Universal Model that Segments Any Bone at Any Location on MRI}, 
      author={Hanxue Gu and Roy Colglazier and Haoyu Dong and Jikai Zhang and Yaqian Chen and Zafer Yildiz and Yuwen Chen and Lin Li and Jichen Yang and Jay Willhite and Alex M. Meyer and Brian Guo and Yashvi Atul Shah and Emily Luo and Shipra Rajput and Sally Kuehn and Clark Bulleit and Kevin A. Wu and Jisoo Lee and Brandon Ramirez and Darui Lu and Jay M. Levin and Maciej A. Mazurowski},
      year={2024},
      eprint={2401.12974},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
```
