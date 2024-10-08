import { useRef, useState } from "react";
import { PhotoMode, PhotoModeToggle } from "./PhotoModeToggle";
import { CameraView } from "expo-camera";
import { StyleSheet } from "react-native";
import { CameraControls } from "./CameraControls";
import { TrainingCategory } from "./TrainingCategory";

export const Camera = () => {
  const [mode, setMode] = useState(PhotoMode.Training);
  const [trainingType, setTrainingType] = useState(TrainingCategory.Control);
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
        trainingType={trainingType}
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
