import {
  CameraPictureOptions,
} from "expo-camera";
import { PhotoMode } from "./PhotoModeToggle";
import { Pressable, View, Text, StyleSheet } from "react-native";
import Toast from "react-native-root-toast";
import { SavePhotoClient } from "./SavePhotoClient";
import { TrainingCategory } from "./TrainingCategory";
import { CameraView } from "expo-camera";
import "setimmediate";
import { EvaluatePhotoClient, EvaluationResults } from "./EvaluatePhotoClient";

const CAMERA_PICTURE_OPTIONS: CameraPictureOptions = {
  quality: 0.25,
  base64: true,
  skipProcessing: false,
  exif: false,
  imageType: "jpg",
};

interface IProps {
  photoMode: PhotoMode;
  cameraRef: React.RefObject<CameraView>;
  trainingType: TrainingCategory;
  toggleTrainingType: () => void;
}

export const CameraControls = (p: IProps) => {

  const uploadPhoto = async (trainingType: TrainingCategory, photoBase64?: string) => {
    const uploadSuccessful = await new SavePhotoClient().uploadPhoto(
      trainingType,
      photoBase64
    );
    try {
      if (uploadSuccessful) showSuccessToast();
      else throw new Error("Failed to upload photo.");
    } catch (error) {
      console.error("Failed to take photo:", error);
      showSavePhotoErrorToast();
    }
  };

  const evaluatePhoto = async (photoBase64?: string) => {
    const evaluationResults = await new EvaluatePhotoClient().evaluatePhoto(
      photoBase64
    );

    try {
      if (evaluationResults)
        showResultsToast(evaluationResults);
      else
        throw new Error("Failed to evaluate photo.");
    } catch (error) {
      console.error("Failed to take photo:", error);
      showEvaluatePhotoErrorToast();
    }
  };

  const takePhoto = async () => {
    if (!p.cameraRef.current) throw new Error("Camera ref is not available.");

    const photo = await p.cameraRef.current.takePictureAsync(
      CAMERA_PICTURE_OPTIONS
    );

    const photoBase64 = sanitizeImage(photo?.base64);

    if (p.photoMode === PhotoMode.Training)
      uploadPhoto(p.trainingType, photoBase64);
    else {
      const results = await evaluatePhoto(photoBase64);
    }
  }

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


function showSavePhotoErrorToast() {
  Toast.show("❌ Failed to Save.", {
    duration: 1500,
    position: Toast.positions.BOTTOM,
    shadow: true,
    animation: true,
    hideOnPress: true,
    backgroundColor: "red",
  });
}

function showEvaluatePhotoErrorToast() {
  Toast.show("❌ Failed to Evaluate Photo.", {
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

const sanitizeImage = (base64Image?: string) =>
  base64Image?.replace("data:image/jpeg;base64,", "");

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


function showResultsToast(results: EvaluationResults) {
  const prediction = getPrediction(results);
  Toast.show(`
        Prediction: ${prediction}\r\n
        Captain: ${results.captain}\r\n 
        B: ${results.bathroom_cat}\r\n, 
        Control: ${results.control}`, {
    duration: 3000,
    position: Toast.positions.BOTTOM,
    shadow: true,
    animation: true,
    hideOnPress: true,
    backgroundColor: "lightgreen",
  });
}

const getPrediction = (results: EvaluationResults) => {
  if (!results) return "No prediction available.";

  const max = Math.max(
    results.captain ?? -1,
    results.bathroom_cat ?? -1,
    results.control ?? -1
  );

  if (max === results.captain) return "😻 Captain";
  if (max === results.bathroom_cat) return "😼 Bathroom Cat";
  if (max === results.control) return "🪑 Control";
  return "No prediction available.";
}