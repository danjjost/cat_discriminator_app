import { useState } from "react";
import { PhotoMode, PhotoModeToggle } from "./PhotoModeToggle";
import { CameraView } from "expo-camera";
import { Pressable, View, StyleSheet, Text } from "react-native";

export const Camera = () => {
  const [mode, setMode] = useState(PhotoMode.Scanning);
  const [trainingType, setTrainingType] = useState(TrainingType.Captain);

  const togglePhotoMode = () => {
    setMode((currentMode) =>
      currentMode === PhotoMode.Scanning
        ? PhotoMode.Training
        : PhotoMode.Scanning
    );
  };

  const toggleTrainingType = () => {
    setTrainingType((currentType) => {
      switch (currentType) {
        case TrainingType.Captain:
          return TrainingType.BathroomCat;
        case TrainingType.BathroomCat:
          return TrainingType.Control;
        case TrainingType.Control:
          return TrainingType.Captain;
      }
    });
  };

  return (
    <CameraView style={styles.camera} facing={"back"}>
      <PhotoModeToggle currentMode={mode} togglePhotoMode={togglePhotoMode} />
      <CameraControls
        photoMode={mode}
        trainingType={trainingType}
        toggleTrainingType={toggleTrainingType}
      />
    </CameraView>
  );
};

enum TrainingType {
  Captain,
  BathroomCat,
  Control,
}

interface ICameraControlProps {
  photoMode: PhotoMode;
  trainingType: TrainingType;
  toggleTrainingType: () => void;
}

function CameraControls(p: ICameraControlProps) {
  const getTrainingType = () => {
    if (p.photoMode != PhotoMode.Training) return "         ";

    switch (p.trainingType) {
      case TrainingType.Captain:
        return "Captain 😻";
      case TrainingType.BathroomCat:
        return "Bathroom Cat 😾";
      case TrainingType.Control:
        return "Control 🪑";
    }
  };

  return (
    <View style={styles.buttonContainer}>
      <Pressable style={styles.button} onPress={p.toggleTrainingType}>
        <Text style={styles.text}>{getTrainingType()}</Text>
      </Pressable>
      <Pressable style={styles.shutterButton}>
        <Text style={styles.shutterButtonText}>{" 📸 "}</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  buttonContainer: {
    flexDirection: "row",
    backgroundColor: "transparent",
    alignContent: "space-around",
    position: "absolute",
    bottom: 0,
    width: "100%",
    justifyContent: "space-around",
    paddingBottom: 50,
  },
  shutterButton: {
    backgroundColor: "rgba(255, 255, 255, 0.75)",
    borderRadius: 100,
    padding: 25,
    alignSelf: "center",
    alignItems: "center",
  },
  shutterButtonText: {
    fontSize: 50,
    marginBottom: 10,
  },
  button: {
    alignSelf: "center",
    alignItems: "center",
  },
  camera: {
    flex: 1,
  },
  text: {
    fontSize: 24,
    fontWeight: "bold",
    color: "white",
  },
});
