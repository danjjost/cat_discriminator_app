import { Config } from "../../config";
import { TrainingCategory } from "../TrainingCategory";

export class SavePhotoClient {
  uploadPhoto = async (
    trainingType: TrainingCategory,
    base64Image?: string
  ): Promise<boolean> => {
    try {
      if (!base64Image) throw new Error("No image data captured!");

      const response = await sendPhotoViaHttp(trainingType, base64Image);

      if (!response.ok)
        throw new Error(`Failed to upload photo: ${response.text()}`);
    } catch (error) {
      console.error("Error uploading photo:", error);
      return false;
    }

    return true;
  };
}

const sendPhotoViaHttp = async (
  trainingCategory: TrainingCategory,
  base64Image: string
) => await fetch(getUrl(trainingCategory), getRequestOptions(base64Image));

const getUrl = (trainingCategory: TrainingCategory): string =>
  `${new Config().upload_url}/${getTrainingCategoryString(trainingCategory)}`;

function getRequestOptions(base64Image: string): RequestInit | undefined {
  return {
    method: "POST",
    headers: {
      "Content-Type": "text/plain",
    },
    body: base64Image,
  };
}

function getTrainingCategoryString(category: TrainingCategory) {
  switch (category) {
    case TrainingCategory.BathroomCat:
      return "bathroom-cat";
    case TrainingCategory.Captain:
      return "captain";
    case TrainingCategory.Control:
      return "control";
  }
}
