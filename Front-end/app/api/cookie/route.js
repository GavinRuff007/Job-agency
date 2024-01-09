// import { cookies } from "next/headers";

// export async function GET() {
//     const cookieStore = cookies();
//     const token = cookieStore.get("token");
//     console.log(token.value);

//     return new Response(`Hello NextJs!: Token -> ${token && token.value}`, {
//         status: 200,
//         headers: { "Set-Cookie": `token=${token}` },
//     });
// }
import { cookies } from "next/headers";

export async function GET() {
  const cookieStore = cookies();
  const token = cookieStore.get("token");

  if (token) {
    const tokenValue = token.value;
    console.log(tokenValue);
  } else {
    console.log("Token doesn't exist");
  }

  return new Response(`Hello NextJs!: Token -> ${token && token.value}`, {
    status: 200,
    headers: { "Set-Cookie": `token=${token}` },
  });
}