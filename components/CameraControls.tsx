import {
  CameraPictureOptions,
  ImageType,
} from "expo-camera/build/legacy/Camera.types";
import { PhotoMode } from "./PhotoModeToggle";
import { Pressable, View, Text, StyleSheet } from "react-native";
import Toast from "react-native-root-toast";
import { SavePhotoClient } from "./SavePhotoClient";
import { TrainingCategory } from "./TrainingCategory";
import { CameraView } from "expo-camera";

const CAMERA_PICTURE_OPTIONS: CameraPictureOptions = {
  quality: 0.25,
  base64: true,
  skipProcessing: false,
  exif: false,
  imageType: ImageType.jpg,
};

interface IProps {
  photoMode: PhotoMode;
  cameraRef: React.RefObject<CameraView>;
  trainingType: TrainingCategory;
  toggleTrainingType: () => void;
}

export const CameraControls = (p: IProps) => {
  const takePhoto = async () => {
    try {
      if (!p.cameraRef.current) throw new Error("Camera ref is not available.");

      const photo = await p.cameraRef.current.takePictureAsync(
        CAMERA_PICTURE_OPTIONS
      );

      var uploadSuccessful = await new SavePhotoClient().uploadPhoto(
        p.trainingType,
        photo?.base64
      );

      if (uploadSuccessful) showSuccessToast();
      else throw new Error("Failed to upload photo.");
    } catch (error) {
      console.error("Failed to take photo:", error);
      showErrorToast();
    }
  };

  return (
    <View style={styles.buttonContainer}>
      <Pressable style={styles.button} onPress={p.toggleTrainingType}>
        <Text style={styles.text}>{getTrainingTypeText(p)}</Text>
      </Pressable>
      <Pressable
        style={styles.shutterButton}
        onPress={() => {
          takePhoto();
        }}
      >
        <Text style={styles.shutterButtonText}>{" 📸 "}</Text>
      </Pressable>
    </View>
  );
};

const getTrainingTypeText = (p: IProps) => {
  if (p.photoMode != PhotoMode.Training) return "         ";

  switch (p.trainingType) {
    case TrainingCategory.Captain:
      return "Captain 😻";
    case TrainingCategory.BathroomCat:
      return "Bathroom Cat 😾";
    case TrainingCategory.Control:
      return "Control 🪑";
  }
};

function showErrorToast() {
  Toast.show("❌ Failed to Save.", {
    duration: 1500,
    position: Toast.positions.BOTTOM,
    shadow: true,
    animation: true,
    hideOnPress: true,
    backgroundColor: "red",
  });
}

function showSuccessToast() {
  Toast.show("✔️ Saved Successfully!", {
    duration: 1500,
    position: Toast.positions.BOTTOM,
    shadow: true,
    animation: true,
    hideOnPress: true,
    backgroundColor: "lightgreen",
  });
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
  text: {
    fontSize: 24,
    fontWeight: "bold",
    color: "white",
  },
});
