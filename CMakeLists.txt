cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SegmentHumanBody)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/mazurowski-lab/SlicerSegmentHumanBody#readme")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Zafer Yildiz (Mazurowski Lab, Duke University)")
set(EXTENSION_DESCRIPTION "SegmentHumanBody aims to asist its users in segmenting MRI medical data on 3D Slicer by integrating varios deep learning models developed by Mazurowski Lab. SegmentHumanBody has CC BY-NC 4.0 license which does not allow commercial use.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/mazurowski-lab/SlicerSegmentHumanBody/main/SegmentHumanBody/Resources/Icons/SegmentHumanBody.png")
set(EXTENSION_SCREENSHOTURLS
  "https://raw.githubusercontent.com/mazurowski-lab/SlicerSegmentHumanBody/main/Screenshots/sws1.png"
  "https://raw.githubusercontent.com/mazurowski-lab/SlicerSegmentHumanBody/main/Screenshots/sws2.png"
  "https://raw.githubusercontent.com/mazurowski-lab/SlicerSegmentHumanBody/main/Screenshots/sws3.png"
)
set(EXTENSION_DEPENDS "PyTorch") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SegmentHumanBody)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
