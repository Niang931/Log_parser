import type { HTMLAttributes } from 'react'

/** 1280px content column with the page gutter: 20px on phones, 80px from xl. */
export function Container({ className = '', ...rest }: HTMLAttributes<HTMLDivElement>) {
  return <div className={`mx-auto w-full max-w-[1440px] px-5 sm:px-8 xl:px-20 ${className}`} {...rest} />
}
