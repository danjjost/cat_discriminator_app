import { Config } from "../config";
import { TrainingCategory } from "./TrainingCategory";

export class SavePhotoClient {
  uploadPhoto = async (
    trainingType: TrainingCategory,
    base64Image?: string
  ): Promise<boolean> => {
    try {
      if (!base64Image) throw new Error("No image data captured!");

      const response = await uploadPhotoViaHttp(trainingType, base64Image);

      if (!response.ok)
        throw new Error(`Failed to upload photo: ${response.text()}`);
    } catch (error) {
      console.error("Error uploading photo:", error);
      return false;
    }

    return true;
  };
}
const uploadPhotoViaHttp = async (
  trainingCategory: TrainingCategory,
  base64Image: string
) =>
  await fetch(
    `${new Config().upload_function_url}/${getCategoryString(
      trainingCategory
    )}`,
    getRequestOptions(base64Image)
  );

function getRequestOptions(base64Image: string): RequestInit | undefined {
  return {
    method: "POST",
    headers: {
      "Content-Type": "text/plain",
    },
    body: sanitizeImage(base64Image),
  };
}

const sanitizeImage = (base64Image: string) =>
  base64Image.replace("data:image/jpeg;base64,", "");

function getCategoryString(category: TrainingCategory) {
  var categoryString = TrainingCategory[category].toLowerCase();

  if (categoryString.includes("bathroom")) return "bathroom-cat";
  else return categoryString;
}
