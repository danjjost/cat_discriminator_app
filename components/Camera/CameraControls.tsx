import { CameraPictureOptions } from "expo-camera";
import { PhotoMode } from "./PhotoModeToggle";
import { Pressable, View, Text, StyleSheet } from "react-native";
import { TrainingCategory } from "../TrainingCategory";
import { CameraView } from "expo-camera";
import "setimmediate";
import { usePhotoEvaluation } from "../hooks/usePhotoEvaluation";
import { usePhotoUpload } from "../hooks/usePhotoUpload";
import { getTrainingCategoryText } from "../utils/getTrainingCategoryText";

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
  trainingCategory: TrainingCategory;
  toggleTrainingType: () => void;
}

export const CameraControls = (p: IProps) => {
  const uploadPhotoAndDisplayResults = usePhotoUpload();
  const evaluatePhotoAndDisplayResults = usePhotoEvaluation();

  const takePhoto = async () => {
    verifyCameraReady(p.cameraRef);

    const photo = await p.cameraRef!.current!.takePictureAsync(
      CAMERA_PICTURE_OPTIONS
    );

    const photoBase64 = sanitizeImage(photo?.base64);

    if (p.photoMode === PhotoMode.Training)
      await uploadPhotoAndDisplayResults(p.trainingCategory, photoBase64);
    else await evaluatePhotoAndDisplayResults(photoBase64);
  };

  return (
    <View style={styles.buttonContainer}>
      <TrainingCategoryButton {...p} />
      <ShutterButton onPress={takePhoto} />
    </View>
  );
};

// ---- Components ----

const TrainingCategoryButton = (p: IProps) => (
  <Pressable style={styles.button} onPress={p.toggleTrainingType}>
    {p.photoMode === PhotoMode.Training && (
      <Text style={styles.text}>
        {getTrainingCategoryText(p.trainingCategory)}
      </Text>
    )}
  </Pressable>
);

const ShutterButton = (p: { onPress: () => void }) => (
  <Pressable
    style={styles.shutterButton}
    onPress={() => {
      p.onPress();
    }}
  >
    <Text style={styles.shutterButtonText}>{" 📸 "}</Text>
  </Pressable>
);

// ---- Utility Functions ----

const sanitizeImage = (base64Image?: string) =>
  base64Image?.replace("data:image/jpeg;base64,", "");

function verifyCameraReady(cameraRef: React.RefObject<CameraView>) {
  if (!cameraRef.current) throw new Error("Camera ref is not available.");
}

// ---- Stylesheet ----

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
