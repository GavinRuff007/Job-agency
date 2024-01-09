import { redirect } from "next/navigation";

const fullname = "mohamad amin mousavi";

export async function GET() {
    redirect(`http://localhost:3000/api/username/${fullname}`);
}
