from string import Template

repo_query = Template("""
{
  rateLimit {
    cost
    remaining
    resetAt
  }
  search(query: "is:public archived:false pushed:>2019-01-01", after: $cursorStart, type: REPOSITORY, first: 50) {
    repositoryCount
    pageInfo {
      hasNextPage
      endCursor
    }
    edges {
      node {
        ... on Repository {
          url
          name
          createdAt
          issues(first: 1, after: null, filterBy: {labels: ["bug", "problem", "Bug", "BUG", "bug report", "Bug report", "Bug Report"]}) {
            totalCount
          }
          defaultBranchRef {
            target {
              ... on Commit {
                history(first: 1) {
                  totalCount
                  edges {
                    node {
                      ... on Commit {
                        committedDate
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
""")

issues_query = """
{
}
"""


full_query="""
{
  rateLimit {
    cost
    remaining
    resetAt
  }
  search(query: "is:public archived:false pushed:>2019-01-01", after: null, type: REPOSITORY, first: 20) {
    repositoryCount
    pageInfo {
      hasNextPage
      startCursor
      endCursor
    }
    edges {
      node {
        ... on Repository {
          url
          name
          createdAt
          issues(first: 10, after: null, filterBy: {labels: ["bug", "problem", "Bug", "BUG"]}) {
            totalCount
            pageInfo {
              hasNextPage
              startCursor
            }
            nodes {
              id
              author {
                login
              }
            }
          }
          defaultBranchRef {
            target {
              ... on Commit {
                history(first: 1) {
                  totalCount
                  edges {
                    node {
                      ... on Commit {
                        committedDate
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

"""
