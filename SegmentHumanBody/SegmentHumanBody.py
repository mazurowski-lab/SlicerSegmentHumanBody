import glob
import os
import pickle
import numpy as np
import json
import qt
import vtk
import shutil
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)
from slicer.util import VTKObservationMixin
import SampleData
from models import cfg
from collections import deque
from PIL import Image

#
# SegmentHumanBody
#
args = cfg.parse_args()
args.if_mask_decoder_adapter=True
args.if_encoder_adapter = True
args.decoder_adapt_depth = 2

class SegmentHumanBody(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SegmentHumanBody"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Zafer Yildiz (Mazurowski Lab, Duke University)"]
        self.parent.helpText = """
The SegmentHumanBody module aims to assist its users in segmenting medical data by integrating
the <a href="https://github.com/facebookresearch/segment-anything">Segment Anything Model (SAM)</a>
developed by Meta.<br>
<br>
See more information in <a href="https://github.com/mazurowski-lab/SlicerSegmentHumanBody">module documentation</a>.
"""
        self.parent.acknowledgementText = """
This file was originally developed by Zafer Yildiz (Mazurowski Lab, Duke University).
"""


#
# SegmentHumanBodyWidget
#


class SegmentHumanBodyWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        global sam_model_registry
        global SamPredictor
        global sab_model_registry
        global seg_any_muscle_pred_one_image
        global sam2_annotation_tool
        global SabPredictor
        global breastModelPredict
        global ct_segmentator
        global torch
        global cv2
        global timm
        global einops
        global gdown
        global nibabel
        global transforms
        global save_image
        

        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self.slicesFolder = self.resourcePath("UI") + "/../../../slices"
        self.featuresFolder = self.resourcePath("UI") + "/../../../features"
        self.annotationMasksFolder = self.resourcePath("UI") + "/../../../annotations"

        self.modelVersion = "vit_t"
        self.modelName = "bone_sam.pth"
        self.modelCheckpoint = self.resourcePath("UI") + "/../../models/" + self.modelName
        self.masks = None
        self.mask_threshold = 0

        slicer.util.warningDisplay(
                "The model and the extension is licensed under the CC BY-NC 4.0 license!\n\nPlease also note that this software is developed for research purposes and is not intended for clinical use yet. Users should exercise caution and are advised against employing it immediately in clinical or medical settings."
            )        
          
        try:
            import PyTorchUtils
        except ModuleNotFoundError:
            raise RuntimeError("You need to install PyTorch extension from the Extensions Manager.")

        minimumTorchVersion = "2.0.0"
        minimumTorchVisionVersion = "0.15.0"
        torchLogic = PyTorchUtils.PyTorchUtilsLogic()

        if not torchLogic.torchInstalled():
            if slicer.util.confirmOkCancelDisplay(
                "PyTorch Python package is required. Would you like to install it now? (it may take several minutes)"
            ):
                torch = torchLogic.installTorch(
                    askConfirmation=True,
                    forceComputationBackend="cu117",
                    torchVersionRequirement=f">={minimumTorchVersion}",
                    torchvisionVersionRequirement=f">={minimumTorchVisionVersion}",
                )
                if torch is None:
                    raise ValueError("You need to install PyTorch to use SegmentHumanBody!")
        else:
            # torch is installed, check version
            from packaging import version

            if version.parse(torchLogic.torch.__version__) < version.parse(minimumTorchVersion):
                raise ValueError(
                    f"PyTorch version {torchLogic.torch.__version__} is not compatible with this module."
                    ' Minimum required version is {minimumTorchVersion}. You can use "PyTorch Util" module'
                    " to install PyTorch with version requirement set to: >={minimumTorchVersion}"
                )

        import torch
        from torchvision import transforms
        from torchvision.utils import save_image
        
        try:
            import timm
            import einops
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "'einops or timm' package is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("einops")
                slicer.util.pip_install("timm")


        try:
            import pandas
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "pandas package is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("pandas")

        try:
            import batchgenerators
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "batchgenerators package is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("batchgenerators")

        try:
            import nnunetv2
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "nnunetv2 package is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("nnunetv2")

        try:
            import pywintypes
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "pypiwin32 package is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("pypiwin32")

        try:
            import blosc2
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "blosc2 package is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("blosc2")

        try:
            import totalsegmentator
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "totalsegmentator package is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("totalsegmentator --user")

        
        try:
            import argparse
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "argparse package is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("argparse")

        try:
            import tensordict
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "tensordict package is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("tensordict")

        try:
            import tensorboard
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "tensorboard package is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("tensorboard")

        try:
            import nibabel
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "nibabel package is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("nibabel")

        try:
            import nrrd
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "pynrrd package is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("pynrrd")

        try:
            import submitit
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "submitit package is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("submitit")
    
        try:
            import hydra
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "hydra package is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("hydra-core")

        try:
            import iopath
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "iopath package is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("iopath")

        try:
            import ruamel.yaml
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "ruamel.yaml package is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("ruamel.yaml")

        try:
            import gdown
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "gdown package is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("gdown")

        try:
            import gdown
        except ModuleNotFoundError:
            raise RuntimeError("There is a problem about the installation of 'gdown' package. Please try again to install!")

        try:
            import git
        except ModuleNotFoundError:
            slicer.util.pip_install("gitpython")

        try: 
            import git
        except ModuleNotFoundError:
            raise RuntimeError("There is a problem about the installation of 'gitpython' package. Please try again to install!")

        os.system("git config --global core.longpaths true")
        
        try: 
            import timm
            import einops
        except ModuleNotFoundError:
            raise RuntimeError("There is a problem about the installation of 'timm' or 'einops' package. Please try again to install!")
        
        copyFolder = self.resourcePath("UI") + "/../../../repo_copy"

        if not os.path.exists(copyFolder):
            os.makedirs(copyFolder)
            git.Repo.clone_from("https://github.com/mazurowski-lab/SlicerSegmentHumanBody", copyFolder)
        
        if not os.path.exists(self.resourcePath("UI") + "/../../models/breast_model"):
            shutil.move(copyFolder + "/SegmentHumanBody/models/breast_model", self.resourcePath("UI") + "/../../models/breast_model")
            #shutil.rmtree(copyFolder, ignore_errors=True)

        if not os.path.exists(self.resourcePath("UI") + "/../../models/breast_model/vnet_with_aug.pth"):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use breast segmentation model? Click OK to install it now!"
            ):
                url = 'https://drive.google.com/uc?id=1IQu_8hYnvAR1_GSKpzkf5l7mqlcQpq3j'
                output = self.resourcePath("UI") + "/../../models/breast_model/vnet_with_aug.pth"
                gdown.download(url, output, quiet=False)

        if not os.path.exists(self.resourcePath("UI") + "/../../models/ct_segmentation"):
            shutil.move(copyFolder + "/SegmentHumanBody/models/ct_segmentation", self.resourcePath("UI") + "/../../models/ct_segmentation")
            #shutil.rmtree(copyFolder, ignore_errors=True)

        if not os.path.exists(self.resourcePath("UI") + "/../../models/ct_segmentation/nnUNetTrainer__nnUNetResEncUNetXLPlans__2d/fold_5/checkpoint_final.pth"):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use CT Segmentation model? Click OK to install it now!"
            ):
                url = 'https://drive.google.com/uc?id=10Pt3nLXxx9nbikhsrTiMH2HQ5p9-GiLB'
                os.makedirs(self.resourcePath("UI") + "/../../models/ct_segmentation/nnUNetTrainer__nnUNetResEncUNetXLPlans__2d/fold_5/", exist_ok=True)
                output = self.resourcePath("UI") + "/../../models/ct_segmentation/nnUNetTrainer__nnUNetResEncUNetXLPlans__2d/fold_5/checkpoint_final.pth"
                gdown.download(url, output, quiet=False)

        if not os.path.exists(self.resourcePath("UI") + "/../../models/sam2_annotation_tool"):
            shutil.move(copyFolder + "/SegmentHumanBody/models/sam2_annotation_tool", self.resourcePath("UI") + "/../../models/sam2_annotation_tool")
            #shutil.rmtree(copyFolder, ignore_errors=True)

        if not os.path.exists(self.resourcePath("UI") + "/../../models/sam2_annotation_tool/sam2_logs/configs/sam2.1_training/lstm_sam2.1_hiera_t.yaml/checkpoints/checkpoint.pt"):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use SLM-SAM2 model? Click OK to install it now!"
            ):
                url = 'https://drive.google.com/uc?id=1uTL1KWjYTIf27_Rs-H5Umr3PogfX1lyB'
                os.makedirs(self.resourcePath("UI") + "/../../models/sam2_annotation_tool/sam2_logs/configs/sam2.1_training/lstm_sam2.1_hiera_t.yaml/checkpoints", exist_ok=True)
                output = self.resourcePath("UI") + "/../../models/sam2_annotation_tool/sam2_logs/configs/sam2.1_training/lstm_sam2.1_hiera_t.yaml/checkpoints/checkpoint.pt"
                gdown.download(url, output, quiet=False)


        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "'segment-anything' is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("https://github.com/facebookresearch/segment-anything/archive/6fdee8f2727f4506cfbbe553e23b895e27956588.zip") 
        try: 
            from segment_anything import sam_model_registry, SamPredictor
            from models.sam import sam_model_registry as sab_model_registry
            from models.segment_any_muscle.main import predict_one_image as seg_any_muscle_pred_one_image
            from models.sam2_annotation_tool.annotation_tool import SAM2AnnotationTool as sam2_annotation_tool
            from models.sam import SamPredictor as SabPredictor
            from models.breast_model.predict_mask_singleimage import breast_model_predict_volume as breastModelPredict
            from models.ct_segmentation import predict_muscle_fat as ct_segmentator
        except ModuleNotFoundError:
            raise RuntimeError("There is a problem about the installation of 'segment-anything' package. Please try again to install!")
        
        try:
            import cv2
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "'open-cv' is missing. Click OK to install it now!"
            ): 
                slicer.util.pip_install("opencv-python")

        try: 
            import cv2
        except ModuleNotFoundError:
            raise RuntimeError("There is a problem about the installation of 'open-cv' package. Please try again to install!")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Working on", self.device)

        self.changeModel("Breast Segmentation Model")
        self.currentlySegmenting = False
        self.featuresAreExtracted = False

    
    def changeModel(self, modelName):
        self.modelName = modelName

        if modelName == "SegmentAnyBone":

            model = sab_model_registry[self.modelVersion](args,checkpoint=self.modelCheckpoint,num_classes=2)
            model.to(device=self.device).eval()
            self.sam = SabPredictor(model)
        
        elif modelName == "SegmentAnyMuscle":
            pass

        elif modelName == "Breast Segmentation Model":
            
            pass

        elif modelName == "SLM-SAM 2":
            self.initializeVariables()

        elif modelName == "CT Segmentation":
            pass

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SegmentHumanBody.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.logic = SegmentHumanBodyLogic()

        # Connections
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.ui.positivePrompts.connect("markupsNodeChanged()", self.updateParameterNodeFromGUI)
        self.ui.positivePrompts.markupsPlaceWidget().setPlaceModePersistency(True)
        self.ui.negativePrompts.connect("markupsNodeChanged()", self.updateParameterNodeFromGUI)
        self.ui.negativePrompts.markupsPlaceWidget().setPlaceModePersistency(True)

        # Buttons
        self.ui.goToSegmentEditorButton.connect("clicked(bool)", self.onGoToSegmentEditor)
        self.ui.goToMarkupsButton.connect("clicked(bool)", self.onGoToMarkups)
        self.ui.runAutomaticSegmentation.connect("clicked(bool)", self.onAutomaticSegmentation)
        self.ui.assignLabel2D.connect('clicked(bool)', self.onAssignLabel2D)
        self.ui.assignLabel3D.connect('clicked(bool)', self.onAssignLabelIn3D)
        #self.ui.startTrainingForSAM2ToolButton.connect('clicked(bool)', self.sam2AnnotationToolTraining)
        self.ui.startInferenceForSAM2ToolButton.connect('clicked(bool)', self.sam2AnnotationToolInference)
        self.ui.segmentButton.connect("clicked(bool)", self.onStartSegmentation)
        self.ui.stopSegmentButton.connect("clicked(bool)", self.onStopSegmentButton)
        self.ui.segmentationDropDown.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.maskDropDown.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.modelDropDown.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)

        self.segmentIdToSegmentationMask = {}
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        #         
        self.setParameterNode(self.logic.getParameterNode())

        if not self._parameterNode.GetNodeReferenceID("positivePromptPointsNode"):
            newPromptPointNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "positive")
            newPromptPointNode.GetDisplayNode().SetSelectedColor(0, 1, 0)
            self._parameterNode.SetNodeReferenceID("positivePromptPointsNode", newPromptPointNode.GetID())

        if not self._parameterNode.GetNodeReferenceID("negativePromptPointsNode"):
            newPromptPointNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "negative")
            newPromptPointNode.GetDisplayNode().SetSelectedColor(1, 0, 0)
            self._parameterNode.SetNodeReferenceID("negativePromptPointsNode", newPromptPointNode.GetID())

        self.ui.positivePrompts.setCurrentNode(self._parameterNode.GetNodeReference("positivePromptPointsNode"))
        self.ui.negativePrompts.setCurrentNode(self._parameterNode.GetNodeReference("negativePromptPointsNode"))

        if not self._parameterNode.GetNodeReferenceID("SAMSegmentationNode"):

            self.samSegmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', 'Segmentation')
            self.samSegmentationNode.CreateDefaultDisplayNodes() 
            self.firstSegmentId = self.samSegmentationNode.GetSegmentation().AddEmptySegment("bone")
            
            self._parameterNode.SetNodeReferenceID("SAMSegmentationNode", self.samSegmentationNode.GetID())
            self._parameterNode.SetParameter("SAMCurrentSegment", self.firstSegmentId)
            self._parameterNode.SetParameter("SAMCurrentMask", "Mask-1")

            self.ui.segmentationDropDown.addItem(self.samSegmentationNode.GetSegmentation().GetNthSegment(0).GetName())
            for i in range(2):
                self.ui.maskDropDown.addItem("Mask-" + str(i+1))

            self.ui.modelDropDown.addItem("SegmentAnyBone")
            self.ui.modelDropDown.addItem("Breast Segmentation Model")
            self.ui.modelDropDown.addItem("SegmentAnyMuscle")
            self.ui.modelDropDown.addItem("CT Segmentation")

            self.ui.ctSegmentationModelDropdown.addItem("Custom")
            self.ui.ctSegmentationModelDropdown.addItem("2D")
            self.ui.ctSegmentationModelDropdown.addItem("3D")
            self.ui.ctSegmentationModelDropdown.addItem("Both")
            

            self.ui.modelDropDown.addItem("SLM-SAM 2")

            
    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        if self._parameterNode is not None and self.hasObserver(
            self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode
        ):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        if not slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode"):
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        if self._parameterNode.GetNodeReferenceID("SAMSegmentationNode"):
            segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode")

            self.ui.segmentationDropDown.clear()
            for i in range(segmentationNode.GetSegmentation().GetNumberOfSegments()):
                segmentName = segmentationNode.GetSegmentation().GetNthSegment(i).GetName()
                self.ui.segmentationDropDown.addItem(segmentName)
 
            if self._parameterNode.GetParameter("SAMCurrentSegment"):
                self.ui.segmentationDropDown.setCurrentText(segmentationNode.GetSegmentation().GetSegment(self._parameterNode.GetParameter("SAMCurrentSegment")).GetName())
            
            if self._parameterNode.GetParameter("SAMCurrentMask"):
                self.ui.maskDropDown.setCurrentText(self._parameterNode.GetParameter("SAMCurrentMask"))

            if self._parameterNode.GetParameter("SAMCurrentModel"):
                self.ui.modelDropDown.setCurrentText(self._parameterNode.GetParameter("SAMCurrentModel"))

        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        if self._parameterNode.GetParameter("SAMCurrentModel") != self.ui.modelDropDown.currentText:
            print(self._parameterNode.GetParameter("SAMCurrentModel"), "=>", self.ui.modelDropDown.currentText)
            self.changeModel(self.ui.modelDropDown.currentText)
            if self.ui.modelDropDown.currentText == "Breast Segmentation Model" or self.ui.modelDropDown.currentText == "SegmentAnyMuscle":
                self.ui.assignLabel2D.hide()
                self.ui.assignLabel3D.hide()
                self.ui.goToMarkupsButton.hide()
                self.ui.segmentButton.hide()
                self.ui.stopSegmentButton.hide()
                self.ui.segmentationDropDown.hide()
                self.ui.maskDropDown.hide()
                self.ui.ctSegmentationModelDropdown.hide()
                self.ui.startTrainingForSAM2ToolButton.hide()
                self.ui.startInferenceForSAM2ToolButton.hide()

            if self.ui.modelDropDown.currentText == "SegmentAnyBone":
                self.ui.assignLabel2D.show()
                self.ui.assignLabel3D.show()
                self.ui.goToMarkupsButton.show()
                self.ui.segmentButton.show()
                self.ui.stopSegmentButton.show()
                self.ui.segmentationDropDown.show()
                self.ui.maskDropDown.show()
                self.ui.ctSegmentationModelDropdown.hide()
                self.ui.startTrainingForSAM2ToolButton.hide()
                self.ui.startInferenceForSAM2ToolButton.hide()


            if self.ui.modelDropDown.currentText == "CT Segmentation":
                self.ui.assignLabel2D.hide()
                self.ui.assignLabel3D.hide()
                self.ui.goToMarkupsButton.hide()
                self.ui.segmentButton.hide()
                self.ui.stopSegmentButton.hide()
                self.ui.segmentationDropDown.hide()
                self.ui.maskDropDown.hide()
                self.ui.ctSegmentationModelDropdown.show()
                self.ui.startTrainingForSAM2ToolButton.hide()
                self.ui.startInferenceForSAM2ToolButton.hide()

            if self.ui.modelDropDown.currentText == "SLM-SAM 2":
                self.ui.assignLabel2D.hide()
                self.ui.assignLabel3D.hide()
                self.ui.goToMarkupsButton.hide()
                self.ui.segmentButton.hide()
                self.ui.stopSegmentButton.hide()
                self.ui.segmentationDropDown.hide()
                self.ui.maskDropDown.hide()
                self.ui.ctSegmentationModelDropdown.show()
                self.ui.startTrainingForSAM2ToolButton.hide()
                self.ui.runAutomaticSegmentation.hide()
                self.ui.ctSegmentationModelDropdown.hide()
                self.ui.startInferenceForSAM2ToolButton.show()

            self._parameterNode.SetParameter("SAMCurrentModel", self.ui.modelDropDown.currentText)

        if not self._parameterNode.GetNodeReference("SAMSegmentationNode") or not hasattr(self, 'volumeShape'):
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode").GetSegmentation()
        self._parameterNode.SetParameter("SAMCurrentSegment", segmentationNode.GetSegmentIdBySegmentName(self.ui.segmentationDropDown.currentText))
        if self._parameterNode.GetParameter("SAMCurrentSegment") not in self.segmentIdToSegmentationMask:
            self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")] = np.zeros(self.volumeShape)

        self._parameterNode.SetParameter("SAMCurrentMask", self.ui.maskDropDown.currentText)
        self._parameterNode.SetParameter("SAMCurrentModel", self.ui.modelDropDown.currentText)

        self._parameterNode.EndModify(wasModified)

    def initializeVariables(self):
        
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())
                self.volume = slicer.util.arrayFromVolume(self._parameterNode.GetNodeReference("InputVolume"))
                self.volumeShape = self.volume.shape
                if self._parameterNode.GetParameter("SAMCurrentSegment") not in self.segmentIdToSegmentationMask:
                    self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")] = np.zeros(self.volumeShape)
                self.sliceAccessorDimension = self.getSliceAccessorDimension()
                sampleInputImage = None

                if self.sliceAccessorDimension == 0:
                    sampleInputImage = self.volume[0,:,:]
                    self.nofSlices = self.volume.shape[0]
                elif self.sliceAccessorDimension == 1:
                    sampleInputImage = self.volume[:,0,:]
                    self.nofSlices = self.volume.shape[1]
                else:
                    sampleInputImage = self.volume[:,:,0]
                    self.nofSlices = self.volume.shape[2]

                self.imageShape = sampleInputImage.shape
            else:
                slicer.util.warningDisplay("You need to add data first to start segmentation!")
                return False
        
        return True
            
    def createSlices(self):
        if not os.path.exists(self.slicesFolder):
            os.makedirs(self.slicesFolder)

        oldSliceFiles = glob.glob(self.slicesFolder + "/*")
        for filename in oldSliceFiles:
            os.remove(filename)

        for sliceIndex in range(self.nofSlices):
            sliceImage = self.getSliceBasedOnSliceAccessorDimension(sliceIndex)
            np.save(self.slicesFolder + "/" + f"slice_{sliceIndex}", sliceImage)

    def createAnnotationMasks(self):
        if not os.path.exists(self.annotationMasksFolder):
            os.makedirs(self.annotationMasksFolder)

        oldSliceFiles = glob.glob(self.annotationMasksFolder + "/*")
        for filename in oldSliceFiles:
            os.remove(filename)


        volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode")
        segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName('Segment_1')
        #print(volumeNode, segmentationNode, segmentId)

        slicer.util.saveNode(volumeNode, self.annotationMasksFolder + "/volume.nii.gz")
        # Get segment as numpy array
        segmentArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, volumeNode)
        
        sliceAnnotationsArr = []

        for sliceIndex in range(self.nofSlices):
            sliceAnnotationImage = self.getAnnotationMaskBasedOnSliceAccessorDimension(segmentArray, sliceIndex)
            sliceAnnotationsArr.append(sliceAnnotationImage)
            np.save(self.annotationMasksFolder + "/" + f"slice_{sliceIndex}", sliceAnnotationImage)
            #im = Image.fromarray(sliceAnnotationImage).convert('RGB')
            #im.save(sliceAnnotationImage, self.annotationMasksFolder + "/" + f"slice_{sliceIndex}_mask.jpeg")

        maskVolume = np.stack(sliceAnnotationsArr, axis=-1)
        #print(volumeNode.shape, maskVolume.shape)
        np.save(self.annotationMasksFolder + "/mask.npy", maskVolume)
        np.save(self.annotationMasksFolder + "/volume.npy", nibabel.load(self.annotationMasksFolder + "/volume.nii.gz").get_fdata())
        

        self.sam2AnnotationTool = sam2_annotation_tool(
            ckpt_folder=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/checkpoints",
            cfg_folder=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/sam2/configs/sam2.1_training",
            search_path=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/configs/sam2.1_training",
        )

    def sam2AnnotationToolInference(self):
        
        self.createAnnotationMasks()
        img_vol_data, mask_vol_data = self.sam2AnnotationTool.load_data(
            img_path=self.annotationMasksFolder + "/volume.npy",
            mask_path=self.annotationMasksFolder + "/mask.npy"
        )

        predictedVolume = self.sam2AnnotationTool.inference(
            data_save_directory=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/example_test_data/",
            img_save_directory=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/example_test_data/images/",
            mask_save_directory=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/example_test_data/masks/",
            volume_name="volume",
            img_vol_data=img_vol_data,
            mask_vol_data=mask_vol_data,
            ann_frame_idx=self.getIndexOfCurrentSlice(),
            checkpoint_folder=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/sam2_logs/configs/sam2.1_training/lstm_sam2.1_hiera_t.yaml/checkpoints",
            cfg_folder=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/sam2/configs/sam2.1"
        )

        for sliceIndex in range(self.getTotalNumberOfSlices()):
            sliceMaskPrediction = predictedVolume[:,:,sliceIndex]
            sliceMask = sliceMaskPrediction.astype(bool) 
            segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode")
            segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName('Segment_1')

            
            if segmentId not in self.segmentIdToSegmentationMask:
                self.segmentIdToSegmentationMask[segmentId] = np.zeros(self.volumeShape)
            if self.sliceAccessorDimension == 2:
                self.segmentIdToSegmentationMask[segmentId][:,:,sliceIndex] = sliceMask
            elif self.sliceAccessorDimension == 1:
                self.segmentIdToSegmentationMask[segmentId][:,sliceIndex,:] = sliceMask
            else:
                self.segmentIdToSegmentationMask[segmentId][sliceIndex,:,:] = sliceMask
                

            slicer.util.updateSegmentBinaryLabelmapFromArray(
                self.segmentIdToSegmentationMask[segmentId],
                self._parameterNode.GetNodeReference("SAMSegmentationNode"),
                segmentId,
                self._parameterNode.GetNodeReference("InputVolume") 
            )


    """def sam2AnnotationToolTraining(self):
        
        img_vol_data, mask_vol_data = self.sam2AnnotationTool.load_data(
            img_path=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/example_train_data/img.npy",
            mask_path=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/example_train_data/mask.npy"
        )

        self.sam2AnnotationTool.train(
            data_save_directory=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/example_train_data/",
            img_save_directory=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/example_train_data/images",
            mask_save_directory=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/example_train_data/masks",
            volume_name="volume",
            img_vol_data=img_vol_data,
            mask_vol_data=mask_vol_data,
            config=self.resourcePath("UI") + "/../../models/sam2_annotation_tool/sam2/configs/sam2.1_training/lstm_sam2.1_hiera_t_mobile.yaml",
            use_cluster=0,
            partition=None,
            account=None,
            qos=None,
            num_gpus=1,
            num_nodes=None
        )
    """


    def getSliceBasedOnSliceAccessorDimension(self, sliceIndex):
        if self.sliceAccessorDimension == 0:
            return self.volume[sliceIndex, :, :]
        elif self.sliceAccessorDimension == 1:
            return self.volume[:, sliceIndex, :]
        else:
            return self.volume[:, :, sliceIndex]
        
    def getAnnotationMaskBasedOnSliceAccessorDimension(self, segmentationArray, sliceIndex):
        if self.sliceAccessorDimension == 0:
            return segmentationArray[sliceIndex, :, :]
        elif self.sliceAccessorDimension == 1:
            return segmentationArray[:, sliceIndex, :]
        else:
            return segmentationArray[:, :, sliceIndex]

    def createFeatures(self):
        if not os.path.exists(self.featuresFolder):
            os.makedirs(self.featuresFolder)

        oldFeatureFiles = glob.glob(self.featuresFolder + "/*")
        for filename in oldFeatureFiles:
            os.remove(filename)

        for filename in os.listdir(self.slicesFolder):
            #if filename in ('slice_68.npy', 'slice_69.npy', 'slice_70.npy'):
            image = np.load(self.slicesFolder + "/" + filename)
            image = (255 * (image - np.min(image)) / np.ptp(image)).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            self.sam.set_image(image)

            with open(self.featuresFolder + "/" + os.path.splitext(filename)[0] + "_features.pkl", "wb") as f:
                pickle.dump(self.sam.features, f)
    
    def onAssignLabelIn3D(self):
        self.initializeSegmentationProcess()

        labelAssigned = False
        promptPointToAssignLabel = None
        nofPositivePromptPoints = self.positivePromptPointsNode.GetNumberOfControlPoints()

        if nofPositivePromptPoints > 0:

            for i in range(nofPositivePromptPoints):
                if self.positivePromptPointsNode.GetNthControlPointVisibility(i):
                    pointRAS = [0, 0, 0]
                    self.positivePromptPointsNode.GetNthControlPointPositionWorld(i, pointRAS)
                    pointIJK = [0, 0, 0, 1]
                    self.volumeRasToIjk.MultiplyPoint(np.append(pointRAS, 1.0), pointIJK)
                    pointIJK = [ int(round(c)) for c in pointIJK[0:3]]

                    if self.sliceAccessorDimension == 2: 
                        promptPointToAssignLabel = [pointIJK[1], pointIJK[2]]
                    elif self.sliceAccessorDimension == 1:
                        promptPointToAssignLabel = [pointIJK[0], pointIJK[2]]
                    elif self.sliceAccessorDimension == 0:
                        promptPointToAssignLabel = [pointIJK[0], pointIJK[1]]
            
            if promptPointToAssignLabel == None and not labelAssigned:
                qt.QTimer.singleShot(100, self.onAssignLabelIn3D)

            currentMask = None
            currentSliceIndex = self.getIndexOfCurrentSlice()
            segmentationIdToBeUpdated = self.getLabelOfPromptPoint(promptPointToAssignLabel)

            sliceIndicesThatContainObject = []
            for sliceIndex in range(self.getTotalNumberOfSlices()):
                if self.sliceAccessorDimension == 2 and self.segmentIdToSegmentationMask[segmentationIdToBeUpdated][:,:,sliceIndex][promptPointToAssignLabel[1], promptPointToAssignLabel[0]]:
                    sliceIndicesThatContainObject.append(sliceIndex)
                elif self.sliceAccessorDimension == 1 and self.segmentIdToSegmentationMask[segmentationIdToBeUpdated][:,sliceIndex,:][promptPointToAssignLabel[1], promptPointToAssignLabel[0]]:
                    sliceIndicesThatContainObject.append(sliceIndex)
                elif self.sliceAccessorDimension == 0 and self.segmentIdToSegmentationMask[segmentationIdToBeUpdated][sliceIndex,:,:][promptPointToAssignLabel[1], promptPointToAssignLabel[0]]:
                    sliceIndicesThatContainObject.append(sliceIndex)
                    
            for currentSliceIndex in sliceIndicesThatContainObject:

                if self.sliceAccessorDimension == 2:
                    currentMask = self.segmentIdToSegmentationMask[segmentationIdToBeUpdated][:,:,currentSliceIndex]
                elif self.sliceAccessorDimension == 1:
                    currentMask = self.segmentIdToSegmentationMask[segmentationIdToBeUpdated][:,currentSliceIndex,:]
                else:
                    currentMask = self.segmentIdToSegmentationMask[segmentationIdToBeUpdated][currentSliceIndex,:,:]

                currentMask = self.bfs(currentMask, promptPointToAssignLabel)

                if self._parameterNode.GetParameter("SAMCurrentSegment") not in self.segmentIdToSegmentationMask:
                    self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")] = np.zeros(self.volumeShape)

                if self.sliceAccessorDimension == 2:
                    self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")][:,:,currentSliceIndex] = currentMask
                    self.segmentIdToSegmentationMask[segmentationIdToBeUpdated][:,:,currentSliceIndex][currentMask == True] = False
                elif self.sliceAccessorDimension == 1:
                    self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")][:,currentSliceIndex,:] = currentMask
                    self.segmentIdToSegmentationMask[segmentationIdToBeUpdated][:,currentSliceIndex,:][currentMask == True] = False
                else:
                    self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")][currentSliceIndex,:,:] = currentMask
                    self.segmentIdToSegmentationMask[segmentationIdToBeUpdated][currentSliceIndex,:,:][currentMask == True] = False
                
                
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                self.segmentIdToSegmentationMask[segmentationIdToBeUpdated],
                self._parameterNode.GetNodeReference("SAMSegmentationNode"),
                segmentationIdToBeUpdated,
                self._parameterNode.GetNodeReference("InputVolume") 
            )

            slicer.util.updateSegmentBinaryLabelmapFromArray(
                self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")],
                self._parameterNode.GetNodeReference("SAMSegmentationNode"),
                self._parameterNode.GetParameter("SAMCurrentSegment"),
                self._parameterNode.GetNodeReference("InputVolume") 
            )

            labelAssigned = True
            for i in range(self.positivePromptPointsNode.GetNumberOfControlPoints()):
                self.positivePromptPointsNode.SetNthControlPointVisibility(i, False)

        if not labelAssigned:
            qt.QTimer.singleShot(100, self.onAssignLabelIn3D)
    
    def getLabelOfPromptPoint(self, promptPoint):
        currentSliceIndex = self.getIndexOfCurrentSlice()

        for segmentId in self.segmentIdToSegmentationMask.keys():
            if self.sliceAccessorDimension == 2 and self.segmentIdToSegmentationMask[segmentId][:,:,currentSliceIndex][promptPoint[1], promptPoint[0]]:
                return segmentId
            elif self.sliceAccessorDimension == 1 and self.segmentIdToSegmentationMask[segmentId][:,currentSliceIndex,:][promptPoint[1], promptPoint[0]]:
                return segmentId
            elif self.sliceAccessorDimension == 0 and self.segmentIdToSegmentationMask[segmentId][currentSliceIndex,:,:][promptPoint[1], promptPoint[0]]:
                return segmentId

    def onAssignLabel2D(self):
        self.initializeSegmentationProcess()

        labelAssigned = False
        promptPointToAssignLabel = None
        nofPositivePromptPoints = self.positivePromptPointsNode.GetNumberOfControlPoints()

        if nofPositivePromptPoints > 0:

            for i in range(nofPositivePromptPoints):
                if self.positivePromptPointsNode.GetNthControlPointVisibility(i):
                    pointRAS = [0, 0, 0]
                    self.positivePromptPointsNode.GetNthControlPointPositionWorld(i, pointRAS)
                    pointIJK = [0, 0, 0, 1]
                    self.volumeRasToIjk.MultiplyPoint(np.append(pointRAS, 1.0), pointIJK)
                    pointIJK = [ int(round(c)) for c in pointIJK[0:3] ]

                    if self.sliceAccessorDimension == 2: 
                        promptPointToAssignLabel = [pointIJK[1], pointIJK[2]]
                    elif self.sliceAccessorDimension == 1:
                        promptPointToAssignLabel = [pointIJK[0], pointIJK[2]]
                    elif self.sliceAccessorDimension == 0:
                        promptPointToAssignLabel = [pointIJK[0], pointIJK[1]]
            
            if promptPointToAssignLabel == None and not labelAssigned:
                qt.QTimer.singleShot(100, self.onAssignLabel2D)

            currentMask = None
            currentSliceIndex = self.getIndexOfCurrentSlice()
            segmentationIdToBeUpdated = self.getLabelOfPromptPoint(promptPointToAssignLabel)

            if self.sliceAccessorDimension == 2:
                currentMask = self.segmentIdToSegmentationMask[segmentationIdToBeUpdated][:,:,currentSliceIndex]
            elif self.sliceAccessorDimension == 1:
                currentMask = self.segmentIdToSegmentationMask[segmentationIdToBeUpdated][:,currentSliceIndex,:]
            else:
                currentMask = self.segmentIdToSegmentationMask[segmentationIdToBeUpdated][currentSliceIndex,:,:]

            currentMask = self.bfs(currentMask, promptPointToAssignLabel)
            
            if self._parameterNode.GetParameter("SAMCurrentSegment") not in self.segmentIdToSegmentationMask:
                self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")] = np.zeros(self.volumeShape)

            if self.sliceAccessorDimension == 2:
                self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")][:,:,currentSliceIndex] = currentMask
                self.segmentIdToSegmentationMask[segmentationIdToBeUpdated][:,:,currentSliceIndex][currentMask == True] = False
            elif self.sliceAccessorDimension == 1:
                self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")][:,currentSliceIndex,:] = currentMask
                self.segmentIdToSegmentationMask[segmentationIdToBeUpdated][:,currentSliceIndex,:][currentMask == True] = False
            else:
                self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")][currentSliceIndex,:,:] = currentMask
                self.segmentIdToSegmentationMask[segmentationIdToBeUpdated][currentSliceIndex,:,:][currentMask == True] = False
            
            
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                self.segmentIdToSegmentationMask[segmentationIdToBeUpdated],
                self._parameterNode.GetNodeReference("SAMSegmentationNode"),
                segmentationIdToBeUpdated,
                self._parameterNode.GetNodeReference("InputVolume") 
            )

            slicer.util.updateSegmentBinaryLabelmapFromArray(
                self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")],
                self._parameterNode.GetNodeReference("SAMSegmentationNode"),
                self._parameterNode.GetParameter("SAMCurrentSegment"),
                self._parameterNode.GetNodeReference("InputVolume") 
            )

            labelAssigned = True
            for i in range(self.positivePromptPointsNode.GetNumberOfControlPoints()):
                self.positivePromptPointsNode.SetNthControlPointVisibility(i, False)

        if not labelAssigned:
            qt.QTimer.singleShot(100, self.onAssignLabel2D)

    def isValidCoordination(self, cRow, cCol, row, col):
        if (cRow >= 0 and cRow < row and cCol >= 0 and cCol < col):
            return True
        return False

    def bfs(self, mask, promptPointToAssignLabel):
        promptPointToAssignLabel = [promptPointToAssignLabel[1], promptPointToAssignLabel[0]]
        visited = np.full(mask.shape, False)
        row, col = mask.shape
        targetValue = mask[promptPointToAssignLabel[0], promptPointToAssignLabel[1]]
        q = deque()
        q.append(promptPointToAssignLabel)
        visited[promptPointToAssignLabel[0]][promptPointToAssignLabel[1]] = True

        while (len(q) > 0):
            temp = q.popleft()
    
            r = temp[0]
            c = temp[1]
    
            if (self.isValidCoordination(r+1, c, row, col) and not visited[r+1][c] and mask[r+1][c] == targetValue):
                q.append([r+1,c])
                visited[r+1][c] = True
            
            if (self.isValidCoordination(r-1, c, row, col) and not visited[r-1][c] and mask[r-1][c] == targetValue):
                q.append([r-1,c])
                visited[r-1][c] = True
            
            if (self.isValidCoordination(r, c-1, row, col) and not visited[r][c-1] and mask[r][c-1] == targetValue):
                q.append([r,c-1])
                visited[r][c-1] = True
            
            if (self.isValidCoordination(r, c+1, row, col) and not visited[r][c+1] and mask[r][c+1] == targetValue):
                q.append([r,c+1])
                visited[r][c+1] = True

        return visited

    def onAutomaticSegmentation(self):
        if not self.initializeVariables():
            return
        
        if self.modelName == "SegmentAnyBone":

            currentSegment = self._parameterNode.GetParameter("SAMCurrentSegment")
            currentSliceIndex = self.getIndexOfCurrentSlice()
            previouslyProducedMask = None

            if self.sliceAccessorDimension == 2:
                previouslyProducedMask = self.segmentIdToSegmentationMask[currentSegment][:, :, currentSliceIndex]
            elif self.sliceAccessorDimension == 1:
                previouslyProducedMask = self.segmentIdToSegmentationMask[currentSegment][:, currentSliceIndex, :]
            else:
                previouslyProducedMask = self.segmentIdToSegmentationMask[currentSegment][currentSliceIndex, :, :]

            if np.any(previouslyProducedMask):
                segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode")
                currentLabel = segmentationNode.GetSegmentation().GetSegment(currentSegment).GetName()

                confirmed = slicer.util.confirmOkCancelDisplay(
                    f"Are you sure you want to re-annotate {currentLabel} for the current slice? All of your previous annotation for {currentLabel} in the current slice will be removed!",
                    windowTitle="Warning",
                )
                if not confirmed:
                    return

            if not self.featuresAreExtracted:
                self.extractFeatures()
                self.featuresAreExtracted = True

            roiList = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
            for roiNode in roiList:
                slicer.mrmlScene.RemoveNode(roiNode)

            for currentSliceIndex in range(self.getTotalNumberOfSlices()):
                with open(self.featuresFolder + "/slice_" + str(currentSliceIndex) + "_features.pkl" , 'rb') as f:
                    self.sam.features = pickle.load(f)

                self.init_masks, _, _ = self.sam.predict(
                    point_coords=None,
                    point_labels=None,
                    box=None,
                    multimask_output=True,
                    return_logits = False,
                )

                self.init_masks = self.init_masks>self.mask_threshold

                if self.init_masks is not None:
                    if self._parameterNode.GetParameter("SAMCurrentMask") == "Mask-1":
                        self.producedMask = self.init_masks[1][:]
                    else:
                        self.producedMask = self.init_masks[0][:]

                else:
                    self.producedMask = np.full(self.sam.original_size, False)
                
                if self._parameterNode.GetParameter("SAMCurrentSegment") not in self.segmentIdToSegmentationMask:
                    self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")] = np.zeros(self.volumeShape)
                
                if self.sliceAccessorDimension == 2:
                    self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")][:,:,currentSliceIndex] = self.producedMask
                elif self.sliceAccessorDimension == 1:
                    self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")][:,currentSliceIndex,:] = self.producedMask
                else:
                    self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")][currentSliceIndex,:,:] = self.producedMask
                    
                self.producedMask = np.full(self.sam.original_size, False)

            slicer.util.updateSegmentBinaryLabelmapFromArray(
                self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")],
                self._parameterNode.GetNodeReference("SAMSegmentationNode"),
                self._parameterNode.GetParameter("SAMCurrentSegment"),
                self._parameterNode.GetNodeReference("InputVolume") 
            )
        
        elif self.modelName == "SegmentAnyMuscle":
            self.muscleSegmentId = self.samSegmentationNode.GetSegmentation().AddEmptySegment("muscle")
            self.createSlices()

            slices = []
            sliceShape = None
            for currentSliceIndex in range(self.getTotalNumberOfSlices()):
                sliceData = np.load(self.slicesFolder + "/" + f"slice_{currentSliceIndex}.npy")
                sliceShape = sliceData.shape
                img = Image.fromarray(sliceData).convert('RGB')
                print(type(img))
                slices.append(img)

            for currentSliceIndex in range(self.getTotalNumberOfSlices()):
                #image_path = 'image.png' # Change this to the image path you want to predict
                #img = Image.open(image_path).convert('RGB')
                img = slices[currentSliceIndex]
                img = transforms.Resize((1024,1024))(img) 
                img = transforms.ToTensor()(img)
                img = (img-img.min())/(img.max()-img.min()+1e-8)
                img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
                img = img.unsqueeze(0)

                # Make prediction
                print(torch.cuda.is_available())
                checkpointLoc = self.resourcePath("UI") + "/../../models/segment_any_muscle/segmentanymuscle.pth"
                mask = seg_any_muscle_pred_one_image(img, checkpointLocation=checkpointLoc)
                print(mask)
                save_image(mask, self.slicesFolder + "/" + f"slice_{currentSliceIndex}_pred.png")
                mask = transforms.Resize((sliceShape[0], sliceShape[1]))(mask.unsqueeze(0))
                print(mask)
                mask = mask.detach().numpy().astype(bool)
                
                print(mask.shape)
                print(type(mask))

                if self.muscleSegmentId not in self.segmentIdToSegmentationMask:
                    self.segmentIdToSegmentationMask[self.muscleSegmentId] = np.zeros(self.volumeShape)

                if self.sliceAccessorDimension == 2:
                    self.segmentIdToSegmentationMask[self.muscleSegmentId][:,:,currentSliceIndex] = mask
                elif self.sliceAccessorDimension == 1:
                    self.segmentIdToSegmentationMask[self.muscleSegmentId][:,currentSliceIndex,:] = mask
                else:
                    self.segmentIdToSegmentationMask[self.muscleSegmentId][currentSliceIndex,:,:] = mask
                    
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                self.segmentIdToSegmentationMask[self.muscleSegmentId],
                self._parameterNode.GetNodeReference("SAMSegmentationNode"),
                self.muscleSegmentId,
                self._parameterNode.GetNodeReference("InputVolume") 
            )

        elif self.modelName == "CT Segmentation":

            volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            inputPath = self.resourcePath("UI") + "/../../models/ct_segmentation/demo" + "/example.nii.gz"
            outputPath = self.resourcePath("UI") + "/../../models/ct_segmentation/results"
            slicer.util.saveNode(volumeNode, inputPath)

            #segmentation, metrics = ct_segmentator.segmentVolume(input_path=inputPath.replace("\\", "/"), body_composition_type="custom", output_path=outputPath.replace("\\", "/"))
            os.system("python C:/Users/zafry/SlicerSegmentHumanBody/SegmentHumanBody/models/ct_segmentation/predict_muscle_fat.py")
            

            slicer.util.loadSegmentation(self.resourcePath("UI") + "/../../models/ct_segmentation/results/example_segmentation.nii.gz")
            
            ctSegmentationNode = slicer.mrmlScene.GetFirstNodeByName('example_segmentation.nii.gz')
            ctSegmentationNode.GetSegmentation().GetNthSegment(0).SetName('muscle')
            ctSegmentationNode.GetSegmentation().GetNthSegment(1).SetName('subcutaneous_fat')
            ctSegmentationNode.GetSegmentation().GetNthSegment(2).SetName('visceral_fat')
            ctSegmentationNode.GetSegmentation().GetNthSegment(3).SetName('muscular_fat')

            with open(outputPath + '/metrics.txt') as f: 
                data = f.read()
            js = json.loads(data)

            confirmed = slicer.util.confirmOkCancelDisplay(
                text=json.dumps(js, indent=4),
                windowTitle="Metrics"
            )
            
        else:
            self.breastTissueSegmentId = self.samSegmentationNode.GetSegmentation().AddEmptySegment("breast tissue")
            self.breastVesselSegmentId = self.samSegmentationNode.GetSegmentation().AddEmptySegment("breast vessel")

            self.createSlices()
            slices = []
            for currentSliceIndex in range(self.getTotalNumberOfSlices()):
                sliceData = np.load(self.slicesFolder + "/" + f"slice_{currentSliceIndex}.npy")
                slices.append(sliceData)

            volume = np.stack(slices, axis=2)
            checkpointPath = self.resourcePath("UI") + "/../../models/breast_model/vnet_with_aug.pth"
            predictedVolume = breastModelPredict(volume, checkpointPath, self.device)

            if self.breastTissueSegmentId not in self.segmentIdToSegmentationMask:
                self.segmentIdToSegmentationMask[self.breastTissueSegmentId] = np.zeros(self.volumeShape)

            if self.breastVesselSegmentId not in self.segmentIdToSegmentationMask:
                self.segmentIdToSegmentationMask[self.breastVesselSegmentId] = np.zeros(self.volumeShape)

            for sliceIndex in range(self.getTotalNumberOfSlices()):
                sliceTissuePrediction = predictedVolume[2,:,:,sliceIndex]
                sliceVesselPrediction = predictedVolume[1,:,:,sliceIndex]

                tissueMask = sliceTissuePrediction.astype(bool) 
                vesselMask = sliceVesselPrediction.astype(bool) 

                if self.sliceAccessorDimension == 2:
                    self.segmentIdToSegmentationMask[self.breastTissueSegmentId][:,:,sliceIndex] = tissueMask
                    self.segmentIdToSegmentationMask[self.breastVesselSegmentId][:,:,sliceIndex] = vesselMask
                elif self.sliceAccessorDimension == 1:
                    self.segmentIdToSegmentationMask[self.breastTissueSegmentId][:,sliceIndex,:] = tissueMask
                    self.segmentIdToSegmentationMask[self.breastVesselSegmentId][:,sliceIndex,:] = vesselMask
                else:
                    self.segmentIdToSegmentationMask[self.breastTissueSegmentId][sliceIndex,:,:] = tissueMask
                    self.segmentIdToSegmentationMask[self.breastVesselSegmentId][sliceIndex,:,:] = vesselMask
                

            slicer.util.updateSegmentBinaryLabelmapFromArray(
                self.segmentIdToSegmentationMask[self.breastTissueSegmentId],
                self._parameterNode.GetNodeReference("SAMSegmentationNode"),
                self.breastTissueSegmentId,
                self._parameterNode.GetNodeReference("InputVolume") 
            )

            slicer.util.updateSegmentBinaryLabelmapFromArray(
                self.segmentIdToSegmentationMask[self.breastVesselSegmentId],
                self._parameterNode.GetNodeReference("SAMSegmentationNode"),
                self.breastVesselSegmentId,
                self._parameterNode.GetNodeReference("InputVolume") 
            )
        
    def onStartSegmentation(self):
        if not self.initializeVariables():
            return

        currentSegment = self._parameterNode.GetParameter("SAMCurrentSegment")
        currentSliceIndex = self.getIndexOfCurrentSlice()
        previouslyProducedMask = None

        if self.sliceAccessorDimension == 2:
            previouslyProducedMask = self.segmentIdToSegmentationMask[currentSegment][:, :, currentSliceIndex]
        elif self.sliceAccessorDimension == 1:
            previouslyProducedMask = self.segmentIdToSegmentationMask[currentSegment][:, currentSliceIndex, :]
        else:
            previouslyProducedMask = self.segmentIdToSegmentationMask[currentSegment][currentSliceIndex, :, :]

        if np.any(previouslyProducedMask):
            segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode")
            currentLabel = segmentationNode.GetSegmentation().GetSegment(currentSegment).GetName()

            confirmed = slicer.util.confirmOkCancelDisplay(
                f"Are you sure you want to re-annotate {currentLabel} for the current slice? All of your previous annotation for {currentLabel} in the current slice will be removed!",
                windowTitle="Warning",
            )
            if not confirmed:
                return

        if not self.featuresAreExtracted:
            self.extractFeatures()
            self.featuresAreExtracted = True

        roiList = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
        for roiNode in roiList:
            slicer.mrmlScene.RemoveNode(roiNode)

        self.currentlySegmenting = True
        self.initializeSegmentationProcess()
        self.collectPromptInputsAndPredictSegmentationMask()
        self.updateSegmentationScene()

    def getSliceAccessorDimension(self):
        npArray = np.zeros((3, 3))
        self._parameterNode.GetNodeReference("InputVolume").GetIJKToRASDirections(npArray)
        npArray = np.transpose(npArray)[0]
        maxIndex = 0
        maxValue = np.abs(npArray[0])

        for index in range(len(npArray)):
            if np.abs(npArray[index]) > maxValue:
                maxValue = np.abs(npArray[index])
                maxIndex = index

        return maxIndex

    def onStopSegmentButton(self):
        self.currentlySegmenting = False
        self.masks = None
        self.init_masks = None

        for i in range(self.positivePromptPointsNode.GetNumberOfControlPoints()):
            self.positivePromptPointsNode.SetNthControlPointVisibility(i, False)

        for i in range(self.negativePromptPointsNode.GetNumberOfControlPoints()):
            self.negativePromptPointsNode.SetNthControlPointVisibility(i, False)

        roiList = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
        for roiNode in roiList:
            slicer.mrmlScene.RemoveNode(roiNode)
        
        

    def onGoToSegmentEditor(self):
        slicer.util.selectModule("SegmentEditor")

    def onGoToMarkups(self):
        slicer.util.selectModule("Markups")

    def getIndexOfCurrentSlice(self):
        redView = slicer.app.layoutManager().sliceWidget("Red")
        redViewLogic = redView.sliceLogic()
        return redViewLogic.GetSliceIndexFromOffset(redViewLogic.GetSliceOffset()) - 1
    
    def getTotalNumberOfSlices(self):
        return self.volume.shape[self.sliceAccessorDimension]

    def updateSegmentationScene(self):
        if self.currentlySegmenting:
            currentSegment = self._parameterNode.GetParameter("SAMCurrentSegment")
            currentSliceIndex = self.getIndexOfCurrentSlice()

            if currentSegment not in self.segmentIdToSegmentationMask:
                self.segmentIdToSegmentationMask[currentSegment] = np.zeros(self.volumeShape)
            if self.sliceAccessorDimension == 2:
                self.segmentIdToSegmentationMask[currentSegment][:, :, currentSliceIndex] = self.producedMask
            elif self.sliceAccessorDimension == 1:
                self.segmentIdToSegmentationMask[currentSegment][:, currentSliceIndex, :] = self.producedMask
            else:
                self.segmentIdToSegmentationMask[currentSegment][currentSliceIndex, :, :] = self.producedMask

            slicer.util.updateSegmentBinaryLabelmapFromArray(
                self.segmentIdToSegmentationMask[currentSegment],
                self._parameterNode.GetNodeReference("SAMSegmentationNode"),
                self._parameterNode.GetParameter("SAMCurrentSegment"),
                self._parameterNode.GetNodeReference("InputVolume"),
            )

        qt.QTimer.singleShot(100, self.updateSegmentationScene)

    def initializeSegmentationProcess(self):
        self.positivePromptPointsNode = self._parameterNode.GetNodeReference("positivePromptPointsNode")
        self.negativePromptPointsNode = self._parameterNode.GetNodeReference("negativePromptPointsNode")

        self.volumeRasToIjk = vtk.vtkMatrix4x4()
        self.volumeIjkToRas = vtk.vtkMatrix4x4()
        self._parameterNode.GetNodeReference("InputVolume").GetRASToIJKMatrix(self.volumeRasToIjk)
        self._parameterNode.GetNodeReference("InputVolume").GetIJKToRASMatrix(self.volumeIjkToRas)

    def combineMultipleMasks(self, masks):
        finalMask = np.full(masks[0].shape, False)
        for mask in masks:
            finalMask[mask == True] = True

        return finalMask

    def collectPromptInputsAndPredictSegmentationMask(self):
        if self.currentlySegmenting:
            self.isTherePromptBoxes = False
            self.isTherePromptPoints = False
            currentSliceIndex = self.getIndexOfCurrentSlice()

            # collect prompt points
            positivePromptPointList, negativePromptPointList = [], []

            nofPositivePromptPoints = self.positivePromptPointsNode.GetNumberOfControlPoints()
            for i in range(nofPositivePromptPoints):
                if self.positivePromptPointsNode.GetNthControlPointVisibility(i):
                    pointRAS = [0, 0, 0]
                    self.positivePromptPointsNode.GetNthControlPointPositionWorld(i, pointRAS)
                    pointIJK = [0, 0, 0, 1]
                    self.volumeRasToIjk.MultiplyPoint(np.append(pointRAS, 1.0), pointIJK)
                    pointIJK = [int(round(c)) for c in pointIJK[0:3]]

                    if self.sliceAccessorDimension == 2:
                        positivePromptPointList.append([pointIJK[1], pointIJK[2]])
                    elif self.sliceAccessorDimension == 1:
                        positivePromptPointList.append([pointIJK[0], pointIJK[2]])
                    elif self.sliceAccessorDimension == 0:
                        positivePromptPointList.append([pointIJK[0], pointIJK[1]])

            nofNegativePromptPoints = self.negativePromptPointsNode.GetNumberOfControlPoints()
            for i in range(nofNegativePromptPoints):
                if self.negativePromptPointsNode.GetNthControlPointVisibility(i):
                    pointRAS = [0, 0, 0]
                    self.negativePromptPointsNode.GetNthControlPointPositionWorld(i, pointRAS)
                    pointIJK = [0, 0, 0, 1]
                    self.volumeRasToIjk.MultiplyPoint(np.append(pointRAS, 1.0), pointIJK)
                    pointIJK = [int(round(c)) for c in pointIJK[0:3]]

                    if self.sliceAccessorDimension == 2:
                        negativePromptPointList.append([pointIJK[1], pointIJK[2]])
                    elif self.sliceAccessorDimension == 1:
                        negativePromptPointList.append([pointIJK[0], pointIJK[2]])
                    elif self.sliceAccessorDimension == 0:
                        negativePromptPointList.append([pointIJK[0], pointIJK[1]])

            promptPointCoordinations = positivePromptPointList + negativePromptPointList
            promptPointLabels = [1] * len(positivePromptPointList) + [0] * len(negativePromptPointList)

            if len(promptPointCoordinations) != 0:
                self.isTherePromptPoints = True

            # collect prompt boxes
            boxList = []

            roiBoxes = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
            
            for roiBox in roiBoxes:
                boxBounds = np.zeros(6)
                roiBox.GetBounds(boxBounds)
                minBoundaries = self.volumeRasToIjk.MultiplyPoint([boxBounds[0], boxBounds[2], boxBounds[4], 1])
                maxBoundaries = self.volumeRasToIjk.MultiplyPoint([boxBounds[1], boxBounds[3], boxBounds[5], 1])
                if self.sliceAccessorDimension == 2:
                    boxList.append([maxBoundaries[1], maxBoundaries[2], minBoundaries[1], minBoundaries[2]])
                elif self.sliceAccessorDimension == 1:
                    boxList.append([maxBoundaries[0], maxBoundaries[2], minBoundaries[0], minBoundaries[2]])
                elif self.sliceAccessorDimension == 0:
                    boxList.append([maxBoundaries[0], maxBoundaries[1], minBoundaries[0], minBoundaries[1]])

            if len(boxList) != 0:
                self.isTherePromptBoxes = True

            # predict mask
            with open(self.featuresFolder + "/" + f"slice_{currentSliceIndex}_features.pkl", "rb") as f:
                self.sam.features = pickle.load(f)

            if self.isTherePromptBoxes and not self.isTherePromptPoints:
                inputBoxes = torch.tensor(boxList, device=self.device)
                transformedBoxes = self.sam.transform.apply_boxes_torch(inputBoxes, self.imageShape)

                self.masks, _, _ = self.sam.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformedBoxes,
                    multimask_output = True,
                    return_logits = False
                )

                self.masks = self.masks.cpu().numpy()
                self.masks = self.combineMultipleMasks(self.masks)

            elif self.isTherePromptPoints and not self.isTherePromptBoxes:
                self.masks, _, _ = self.sam.predict(
                    point_coords=np.array(promptPointCoordinations),
                    point_labels=np.array(promptPointLabels),
                    multimask_output=True,
                    return_logits = False,
                )
                

            elif self.isTherePromptBoxes and self.isTherePromptPoints:
                self.masks, _, _ = self.sam.predict(
                    point_coords=np.array(promptPointCoordinations),
                    point_labels=np.array(promptPointLabels),
                    box=np.array(boxList[0]),
                    multimask_output=True,
                    return_logits = False,
                )

            else:
                self.masks = None

            if self.masks is not None:
                self.masks = self.masks>self.mask_threshold
            if self.masks is not None:
                if self._parameterNode.GetParameter("SAMCurrentMask") == "Mask-1":
                    self.producedMask = self.masks[1][:]
                elif self._parameterNode.GetParameter("SAMCurrentMask") == "Mask-2":
                    self.producedMask = self.masks[0][:]
            else:
                self.producedMask = np.full(self.sam.original_size, False)

            qt.QTimer.singleShot(100, self.collectPromptInputsAndPredictSegmentationMask)

    def extractFeatures(self):
        with slicer.util.MessageDialog("Please wait until SAM has processed the input."):
            with slicer.util.WaitCursor():
                self.createSlices()
                self.createFeatures()

        print("Features are extracted. You can start segmentation by placing prompt points or ROIs (boundary boxes)!")


#
# SegmentHumanBodyLogic
#


class SegmentHumanBodyLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")
