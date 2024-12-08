import { Config } from "../config";
import { TrainingCategory } from "./TrainingCategory";

export class EvaluationResults
{
    bathroom_cat?: number;
    captain?: number;
    control?: number;
}

export class EvaluatePhotoClient {
  evaluatePhoto = async (
    base64Image?: string
  ): Promise<EvaluationResults | null> => {
    try {
      if (!base64Image) throw new Error("No image data captured!");

      console.log(new Config().evaluate_url)
      const response = await evaluatePhotoViaHttp(base64Image);

      if (!response.ok){
        throw new Error(`Failed to upload photo: ${response.text()}`);
      }

      return JSON.parse(await response.text()) as EvaluationResults;
    } catch (error) {
      console.error(JSON.stringify(error));
      console.error("Error uploading photo:", error);
      return null;
    }
  };
}
const evaluatePhotoViaHttp = async (
  base64Image: string
) =>
  await fetch(
    new Config().evaluate_url??"",
    getRequest(base64Image)
  );

function getRequest(base64Image: string): RequestInit | undefined {
  return {
    method: "POST",
    headers: {
      "Content-Type": "text/plain",
    },
    body: base64Image
  };
}
