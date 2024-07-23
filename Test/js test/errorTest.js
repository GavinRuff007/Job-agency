import React from "react";
import { Error } from "./error";

describe("Error", () => {
  it("should log the error to the console", () => {
    const error = {
      message: "Something went wrong",
      stack: "Stack trace",
    };

    const component = mount(<Error error={error} />);

    expect(console.log.mock.calls[0][0]).toEqual(error);
  });
});