#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

const size_t BMP_HEADER_SIZE = 54;

const size_t BMP_HEADER_WIDTH_OFFSET = 18;
const size_t BMP_HEADER_HEIGHT_OFFSET = 22;

const size_t PIXEL_REAL_SIZE = 3;

size_t getWidth(const unsigned char * header)
{
	const unsigned char * start_position = header + BMP_HEADER_WIDTH_OFFSET;
	unsigned int value = *((const unsigned int*)start_position);
	return (size_t)value;
}

size_t getHeight(const unsigned char * header)
{
	const unsigned char * start_position = header + BMP_HEADER_HEIGHT_OFFSET;
	unsigned int value = *((const unsigned int*)start_position);
	return (size_t)value;
}

struct RgbPixel
{
	unsigned char b;
	unsigned char g;
	unsigned char r;
};

struct RgbPixel * allocateMemory(size_t size)
{
	struct RgbPixel * data = (struct RgbPixel *)malloc(sizeof(struct RgbPixel) * size);
	return data;
}

void deallocateMemory(struct RgbPixel ** data)
{
	free(*data);
	*data = NULL;
}

bool readBmp(
	const char * filename,
	unsigned char * header,
	struct RgbPixel ** bmp
)
{
	size_t read;

	FILE * fileBmp = fopen(filename, "rb");
	if (!fileBmp)
		return false;

	read = fread(header, sizeof(unsigned char), BMP_HEADER_SIZE, fileBmp);
	if (read != BMP_HEADER_SIZE)
	{
		fclose(fileBmp);
		return false;
	}

	size_t width = getWidth(header), height = getHeight(header);
	size_t size = width * height;

	*bmp = allocateMemory(size);
	for (size_t i = 0; i < size; i++)
	{
		read = fread((*bmp) + i, sizeof(unsigned char), PIXEL_REAL_SIZE, fileBmp);
		if (read != PIXEL_REAL_SIZE)
		{
			deallocateMemory(bmp);
			fclose(fileBmp);
			return false;
		}
	}

	fclose(fileBmp);
	return true;
}

bool writeBmp(
	const char * filename,
	const unsigned char * header,
	struct RgbPixel * bmp
)
{
	FILE * bmpFile = fopen(filename, "wb");
	if (!bmpFile)
	{
		fclose(bmpFile);
		return false;
	}

	fwrite(header, sizeof(unsigned char), BMP_HEADER_SIZE, bmpFile);
	size_t width = getWidth(header), height = getHeight(header);
	size_t size = width * height;

	for (size_t i = 0; i < size; i++)
		fwrite(bmp + i, sizeof(unsigned char), PIXEL_REAL_SIZE, bmpFile);

	fclose(bmpFile);
	return true;
}

void processImageSmoothFilter(struct RgbPixel * bmp, size_t width, size_t height, size_t radius, struct RgbPixel * result)
{
#define ROUND(x) (unsigned char)((x) + 0.5)

	#pragma omp parallel for shared(bmp, width, height, radius, result) num_threads(4)
	for (size_t i = 0; i < height; i++)
		for (size_t j = 0; j < width; j++)
		{
			unsigned int r = 0, g = 0, b = 0;

			size_t
				starti = i < radius ? 0 : i - radius,
				endi = i + radius >= height ? height - 1 : i + radius,
				startj = j < radius ? 0 : j - radius,
				endj = j + radius >= width ? width - 1 : j + radius;

			size_t count = (endi - starti + 1) * (endj - startj + 1);
			for (size_t ii = starti; ii <= endi; ii++)
				for (size_t jj = startj; jj <= endj; jj++)
				{
					size_t position = ii * width + jj;
					r += bmp[position].r;
					g += bmp[position].g;
					b += bmp[position].b;
				}

			size_t resultPosition = i * width + j;
			result[resultPosition].r = ROUND((float)r / count);
			result[resultPosition].g = ROUND((float)g / count);
			result[resultPosition].b = ROUND((float)b / count);
		}
#undef ROUND
}

int main(int argc, char * argv[])
{
	if (argc != 4)
	{
		printf("%s\n", "Usage: <program name> <input bmp file name> <output bmp file name> <radius>");
		return 0;
	}

	const char * inputBmpFileName = argv[1];
	const char * outputBmpFileName = argv[2];
	size_t radius = (size_t)atoi(argv[3]);

	unsigned char header[BMP_HEADER_SIZE];
	struct RgbPixel * bmp = NULL;

	time_t totalBegin = time(NULL);

	bool readResult = readBmp(inputBmpFileName, header, &bmp);
	if (!readResult)
	{
		printf("Cannot read bmp file %s\n", inputBmpFileName);
		return 1;
	}

	size_t width = getWidth(header), height = getHeight(header);
	struct RgbPixel * resultBmp = allocateMemory(width * height);

	time_t begin = time(NULL);
	for (int i = 0; i < 100; i++)
		processImageSmoothFilter(bmp, width, height, radius, resultBmp);
	time_t end = time(NULL);

	bool writeResult = writeBmp(outputBmpFileName, header, resultBmp);
	if (!writeResult)
	{
		printf("Cannot write bmp file %s\n", outputBmpFileName);
		return 1;
	}

	deallocateMemory(&bmp);
	deallocateMemory(&resultBmp);

	time_t totalEnd = time(NULL);

	printf("Algorithm time: %.2f sec.\n", (double)(end - begin));
	printf("Total time: %.2f sec.\n", (double)(totalEnd - totalBegin));

	return 0;
}

