import { useRef, useState } from "react";
import { PhotoModeToggle } from "./PhotoModeToggle";
import { CameraView } from "expo-camera";
import { StyleSheet } from "react-native";
import { TrainingCategory } from "../models/TrainingCategory";
import { CameraControls } from "./CameraControls";
import { PhotoMode } from "../models/PhotoMode";

export const Camera = () => {
  const [mode, setMode] = useState(PhotoMode.Training);
  const [trainingCategory, setTrainingType] = useState(
    TrainingCategory.Control
  );
  const cameraRef = useRef<CameraView>(null);

  const togglePhotoMode = () => {
    setMode((currentMode) => getNextPhotoMode(currentMode));
  };

  const toggleTrainingType = () => {
    setTrainingType((currentType) => getNextTrainingType(currentType));
  };

  return (
    <CameraView
      mute={true}
      style={styles.camera}
      facing={"back"}
      ref={cameraRef}
    >
      <PhotoModeToggle currentMode={mode} togglePhotoMode={togglePhotoMode} />
      <CameraControls
        cameraRef={cameraRef}
        photoMode={mode}
        trainingCategory={trainingCategory}
        toggleTrainingType={toggleTrainingType}
      />
    </CameraView>
  );
};

const styles = StyleSheet.create({
  camera: {
    flex: 1,
  },
});

const getNextTrainingType = (currentType: TrainingCategory) => {
  switch (currentType) {
    case TrainingCategory.Captain:
      return TrainingCategory.BathroomCat;
    case TrainingCategory.BathroomCat:
      return TrainingCategory.Control;
    case TrainingCategory.Control:
      return TrainingCategory.Captain;
  }
};

const getNextPhotoMode = (currentMode: PhotoMode): PhotoMode => {
  return currentMode === PhotoMode.Scanning
    ? PhotoMode.Training
    : PhotoMode.Scanning;
};
