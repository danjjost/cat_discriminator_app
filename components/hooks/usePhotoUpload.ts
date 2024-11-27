import Toast from "react-native-root-toast";
import { SavePhotoClient } from "../network/SavePhotoClient";
import { TrainingCategory } from "../models/TrainingCategory";

export function usePhotoUpload() {
  return uploadPhoto;
}

async function uploadPhoto(
  trainingType: TrainingCategory,
  photoBase64?: string
) {
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
}

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
