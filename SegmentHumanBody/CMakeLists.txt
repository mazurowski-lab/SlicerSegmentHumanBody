#-----------------------------------------------------------------------------
set(MODULE_NAME SegmentHumanBody)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  models/cfg.py
  models/sam/predictor.py
  models/sam/build_sam.py
  models/sam/__init__.py
  models/sam/automatic_mask_generator.py
  models/sam/modeling/__init__.py
  models/sam/modeling/common.py
  models/sam/modeling/image_encoder.py
  models/sam/modeling/mask_decoder.py
  models/sam/modeling/prompt_encoder.py
  models/sam/modeling/sam.py
  models/sam/modeling/tiny_vit_sam.py
  models/sam/modeling/transformer.py
  models/sam/modeling/vit.py
  models/sam/utils/__init__.py
  models/sam/utils/amg.py
  models/sam/utils/onnx.py
  models/sam/utils/transforms.py
  models/bone_sam.pth
  )
set(EXTENSION_HOMEPAGE "https://github.com/mazurowski-lab/SlicerSegmentHumanBody")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Zafer Yildiz (Duke University - Mazurowski Lab)")
set(EXTENSION_DESCRIPTION "SegmentHumanBody aims to asist its users in segmenting medical data on 3D Slicer by comprehensively integrating the SegmentAnyBone developed by Mazurowski Lab. SegmentHumanBody has CC BY-NC 4.0 license which does not allow commercial use.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/mazurowski-lab/SlicerSegmentHumanBody/SegmentHumanBody/Resources/Icons/SegmentHumanBody.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/mazurowski-lab/SlicerSegmentHumanBody/Screenshots/sws1.png")
set(EXTENSION_DEPENDS "PyTorch") # Specified as a list or "NA" if no dependencies
#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
