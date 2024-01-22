// /** @type {import('next').NextConfig} */
// const nextConfig = {
//   "images": {
//     "remotePatterns": [
//       {
//         "hostname": "platformboy.com",
//         "protocol": "https",
//         "port": "",
//         "pathname": "wp-content/uploads/2022/10/**"
//       },
//       {
//         "hostname": "platformboy.com",
//         "pathname": "https://platformboy.com/wp-content/uploads/2022/10/%D8%B9%DA%A9%D8%B3-%D9%BE%D8%B1%D9%88%D9%81%D8%A7%DB%8C%D9%84-%D9%BE%D8%B3%D8%B1-%D8%B3%D8%A8%D8%B2%D9%87-%D9%85%D9%88-%D9%85%D8%B4%DA%9A%DB%8C.png",
//       }
//     ]
//   }
// };

// module.exports = nextConfig/** @type {import('next').NextConfig} */
const nextConfig = {
    images: {
      domains: ["platformboy.com"],
    },
  };
  
  module.exports = nextConfig;
  