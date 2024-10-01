//get from env variables
export class Config {
  upload_function_url?: string = process.env.UPLOAD_FUNCTION_URL;
  evaluate_function_url?: string = process.env.REACT_APP_EVALUATE_URL;
}
