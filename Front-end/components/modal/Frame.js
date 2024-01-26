import Image from 'next/image';

export default function Frame({ photo }) {
  return (
    <>
      <div className="blur-container">
        <Image
          alt=""
          src={photo}
          height={700}
          width={700}
          className="w-full object-cover aspect-square col-span-2"
        />
      </div>
    </>
  );
}