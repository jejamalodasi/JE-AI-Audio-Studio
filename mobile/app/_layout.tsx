import { Stack } from "expo-router";

export default function RootLayout() {
  return (
    <Stack
      screenOptions={{
        headerStyle: { backgroundColor: "#0b0d12" },
        headerTintColor: "#f5f7fb",
        headerShadowVisible: false,
        contentStyle: { backgroundColor: "#0b0d12" },
      }}
    >
      <Stack.Screen name="index" options={{ title: "JE AI Audio Studio" }} />
    </Stack>
  );
}
