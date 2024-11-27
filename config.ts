//get from env variables
export class Config {
  upload_url?: string = process.env.EXPO_PUBLIC_UPLOAD_FUNCTION_URL;
  evaluate_url?: string = process.env.EXPO_PUBLIC_EVALUATE_URL;
}
