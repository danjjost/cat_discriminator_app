import Toast, { ToastOptions } from "react-native-root-toast";
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
    if (evaluationResults) {
      showResultsToasts(evaluationResults);
    } else {
      throw new Error("Failed to evaluate photo.");
    }
  } catch (error) {
    console.error("Failed to take photo:", error);
    showPhotoEvaluationErrorToast();
  }

  return evaluationResults;
}

// ---- Toast Functions ----

function showResultsToasts(results: EvaluationResults) {
  const prediction = getPrediction(results);

  showToast(0, "Prediction: " + prediction, "lightblue");
  showPercentageToast(1, "😻", results.captain);
  showPercentageToast(2, "😼", results.bathroom_cat);
  showPercentageToast(3, "🪑", results.control);
}

const showToast = (
  position: 3 | 2 | 1 | 0,
  message: string,
  colorOverride?: string
) => Toast.show(message, getToastOptions(position, colorOverride));

const showPercentageToast = (
  position: 3 | 2 | 1 | 0,
  title: string,
  confidenceAsDecimalBetween0And1: number | undefined
) =>
  showToast(
    position,
    `${title} ${toPercent(confidenceAsDecimalBetween0And1)}%`,
    getColorFromConfidence(confidenceAsDecimalBetween0And1 ?? 0)
  );

const showPhotoEvaluationErrorToast = () =>
  Toast.show("❌ Failed to Evaluate Photo.", errorToastOptions);

// ---- Toast Options ----

const getToastOptions = (
  position: 3 | 2 | 1 | 0,
  color?: string
): ToastOptions => ({
  position: getPositionOffset(position),
  backgroundColor: color,
  ...defaultToastOptions,
});

const defaultToastOptions = {
  duration: 3000,
  shadow: true,
  animation: true,
  hideOnPress: false,
};
const errorToastOptions = {
  position: Toast.positions.TOP,
  backgroundColor: "red",
  ...defaultToastOptions,
};

// ---- Helper Functions ----

const getPrediction = (results: EvaluationResults) => {
  if (!results) return "No prediction available.";

  const max = Math.max(
    results.captain ?? 0,
    results.bathroom_cat ?? 0,
    results.control ?? 0
  );

  if (max === results.captain) return "😻";
  if (max === results.bathroom_cat) return "😼";
  if (max === results.control) return "🪑";
  return "No prediction available.";
};

const getColorFromConfidence = (value: number): string | undefined => {
  const red = Math.floor(255 * (1 - value));
  const green = Math.floor(255 * value);

  return `rgb(${red}, ${green}, 0)`;
};

const toPercent = (captain: number | undefined) =>
  captain ? Math.round(captain * 100) : 0;

const getPositionOffset = (value: number) => Toast.positions.TOP + 50 * value;
