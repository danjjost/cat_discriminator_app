import { Pressable, StyleSheet, Text, View } from "react-native";

export enum PhotoMode {
  Scanning,
  Training,
}

interface IProps {
  currentMode: PhotoMode;
  togglePhotoMode: () => void;
}

export const PhotoModeToggle = (p: IProps) => (
  <View style={styles.modeContainer}>
    <Pressable onPress={p.togglePhotoMode} style={getStyles(p)}>
      <Text style={styles.buttonFont}>{getButtonText(p)}</Text>
    </Pressable>
  </View>
);

const getButtonText = (p: IProps) => {
  return p.currentMode === PhotoMode.Scanning
    ? "Scanning Mode 📸"
    : "Training Mode 🎓";
};

const getStyles = (p: IProps) => {
  return p.currentMode === PhotoMode.Scanning
    ? styles.scanningButton
    : styles.trainingButton;
};

const styles = StyleSheet.create({
  modeContainer: {
    padding: 20,
    paddingTop: 50,
    alignItems: "center",
  },
  trainingButton: {
    backgroundColor: "lightgreen",
    padding: 10,
    borderRadius: 10,
    alignSelf: "flex-start",
  },
  scanningButton: {
    backgroundColor: "lightblue",
    padding: 10,
    borderRadius: 10,
    alignSelf: "flex-start",
  },
  buttonFont: {
    fontSize: 20,
  },
});
