import Toast from "react-native-root-toast";
import {
  EvaluatePhotoClient,
  EvaluationResults,
} from "../network/EvaluatePhotoClient";

export function usePhotoEvaluation() {
  return evaluatePhotoAndDisplayResults;
}

async function evaluatePhotoAndDisplayResults(photoBase64?: string) {
  const evaluationResults = await new EvaluatePhotoClient().evaluatePhoto(
    photoBase64
  );

  try {
    if (evaluationResults) showResultsToast(evaluationResults);
    else throw new Error("Failed to evaluate photo.");
  } catch (error) {
    console.error("Failed to take photo:", error);
    showEvaluatePhotoErrorToast();
  }

  return evaluationResults;
}

function showResultsToast(results: EvaluationResults) {
  const prediction = getPrediction(results);

  Toast.show(
    `
          Prediction: ${prediction}\r\n
          Captain: ${results.captain}\r\n 
          B: ${results.bathroom_cat}\r\n, 
          Control: ${results.control}`,
    {
      duration: 3000,
      position: Toast.positions.BOTTOM,
      shadow: true,
      animation: true,
      hideOnPress: true,
      backgroundColor: "lightgreen",
    }
  );
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
};
