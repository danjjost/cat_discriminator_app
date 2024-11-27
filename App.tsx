import { useCameraPermissions } from "expo-camera";
import { Button, StyleSheet, Text, View } from "react-native";
import { Camera } from "./components/Camera/Camera";
import React from "react";
import { RootSiblingParent } from "react-native-root-siblings";
import Toast from "react-native-root-toast";

export default function App() {
  return (
    <>
      <PermissionGatedApp />
      <Toast />
    </>
  );
}

const PermissionGatedApp = () => {
  const [permission, requestPermission] = useCameraPermissions();

  if (!permission) {
    return <View />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>
          Hey dummy, you need to give the app camera permissions:
        </Text>
        <Button onPress={requestPermission} title="grant permission" />
      </View>
    );
  }
  return (
    <RootSiblingParent>
      <View style={styles.container}>
        <Camera />
      </View>
    </RootSiblingParent>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
  },
  message: {
    textAlign: "center",
    paddingBottom: 10,
  },
});
