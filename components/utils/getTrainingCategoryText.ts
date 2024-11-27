import { TrainingCategory } from "../TrainingCategory";

export const getTrainingCategoryText = (trainingCategory: TrainingCategory) => {
  switch (trainingCategory) {
    case TrainingCategory.Captain:
      return "Captain 😻";
    case TrainingCategory.BathroomCat:
      return "Bathroom Cat 😾";
    case TrainingCategory.Control:
      return "Control 🪑";
  }
};
