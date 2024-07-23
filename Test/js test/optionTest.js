import { NextAuthOptions } from "next-auth";
import { expect } from "@jest/globals";

jest.mock("next-auth/providers/github"); // Mock external provider for isolation

describe("options", () => {
  it("should configure providers correctly", () => {
    const options: NextAuthOptions = require("./path/to/options"); // Import your options file

    expect(options.providers).toHaveLength(2);
    expect(options.providers[0].name).toBe("Github");
    expect(options.providers[1].name).toBe("Credentials");
  });

  it("should configure CredentialsProvider credentials", () => {
    const options: NextAuthOptions = require("./path/to/options");

    const credentials = options.providers[1].credentials;
    expect(credentials.username.label).toBe("نام کاربری");
    expect(credentials.password.type).toBe("password");
  });

  it("should authorize correct credentials", async () => {
    const options: NextAuthOptions = require("./path/to/options");

    const user = await options.providers[1].authorize({
      username: "Mohammad",
      password: "17513263",
    });

    expect(user).toEqual({
      id: "11",
      name: "محمدامین موسوی",
      username: "Mohammad",
      password: "17513263",
    });
  });

  it("should reject invalid credentials", async () => {
    const options: NextAuthOptions = require("./path/to/options");

    const user = await options.providers[1].authorize({
      username: "wrong",
      password: "incorrect",
    });

    expect(user).toBeNull();
  });
});
