import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import LoginPage from "./LoginPage";
import { signIn } from "next-auth/react";

jest.mock("next-auth/react"); // Mock NextAuth for isolation

// Test form submission with valid credentials
test("should handle successful login with credentials", async () => {
  signIn.mockResolvedValueOnce({ error: null }); // Mock successful login

  render(<LoginPage />);

  const usernameInput = screen.getByLabelText("نام کاربری");
  const passwordInput = screen.getByLabelText("کلمه عبور");
  const submitButton = screen.getByRole("button", { name: /ورود/i });

  userEvent.type(usernameInput, "test-user");
  userEvent.type(passwordInput, "test-password");
  userEvent.click(submitButton);

  await waitFor(() => {
    expect(signIn).toHaveBeenCalledWith("credentials", {
      username: "test-user",
      password: "test-password",
      redirect: false,
      callbackUrl: "/quiz",
    });
    expect(router.push).toHaveBeenCalledWith("/quiz");
  });
});

// Test form submission with invalid credentials
test("should handle failed login with credentials", async () => {
  signIn.mockRejectedValueOnce(new Error("Invalid credentials"));

  render(<LoginPage />);

  // ... (similar to previous test, but asserting error message)
});

// Test GitHub login button
test("should redirect to GitHub login", async () => {
  render(<LoginPage />);

  const githubButton = screen.getByRole("button", { name: /ورود با گیت هاب/i });
  userEvent.click(githubButton);

  await waitFor(() => {
    expect(signIn).toHaveBeenCalledWith("github", { callbackUrl: "/quiz" });
  });
});
